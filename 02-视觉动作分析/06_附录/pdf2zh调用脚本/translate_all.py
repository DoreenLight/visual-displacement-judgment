#!/usr/bin/env python3
"""用 MiniMax-M3 把 MinerU 英文抽取全文翻译为中文。

改进点（相对 pdf2zh/translate_md.py）：
  1. 分块保持 HTML <table> / ``` 代码块 / 公式块 完整，不跨块切断
  2. 每块结果写 JSON checkpoint，失败/中断可断点续跑
  3. 输出校验：中文占比过低视为「模型没翻」，重试该块
  4. 产物落 <论文目录>/<论文名>_全文翻译.md，并把 images/ 同步到论文目录

用法：
    python translate_all.py                # 全 29 篇
    python translate_all.py --only "Li2024" # 只跑名字里含 Li2024 的
    python translate_all.py --dry-run      # 只报计划不调用 API
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, "/Users/xbpd/Projects/pdf2zh")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "paper_manifest", Path(__file__).resolve().parent / "paper_manifest.py")
_manifest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_manifest)

PDF2ZH_ROOT = Path("/Users/xbpd/Projects/pdf2zh")
CACHE_DIR = Path("/tmp/zh_translate_cache")
MODEL = "MiniMax-M3"
BASE_URL = "https://api.minimaxi.com/v1"

# ---------------------------------------------------------------- system prompt
SYSTEM_PROMPT = """你是资深学术译者，把用户给的英文论文 Markdown 完整翻译成中文。硬性规则：

【必须保留、逐字不动】
- 行内公式 $...$ 与块级公式 $$...$$：内部每一个字符都不动（含变量下标、\\begin{}/\\end{}、\\frac 等 LaTeX 命令）。
- 代码块 ```lang ... ``` 整块原样输出，内部代码与注释都不翻译。
- 图片引用 ![](images/xxx.jpg) 与 <img> 标签：原样输出，alt 文本保持英文。
- URL、DOI、arXiv 编号、邮箱原样。
- 正文中的文献引用标记 [1] [12, 34] 或上标编号原样，不要改写编号。

【需要翻译】
- 全部散文正文：标题、摘要、引言、方法、实验、结果、讨论、结论、致谢。
- 表格：<table>/<tr>/<td> 结构保持不变，只把单元格里的英文文字换成中文；表题（Table 1: ...）要译。
- 图题（Figure N. ...）要译。
- 专有名词首次出现时写成「中文译名（English Original）」，之后统一用中文或统一用英文，不要来回跳。
- 人名保留英文原名（可用「张伟（Wei Zhang）」形式）。
- 数据集名、模型名、机构名保留英文原文：EMDB、3DPW、OpenSim、SMPL、Stanford、OpenCap 等。

【参考文献章节】
References / Bibliography / 参考文献 整节都要翻译成中文：
- 作者名、期刊名、年份、卷期页码、DOI 保持英文原样，不要音译。
- 文章标题翻译成中文。
- 出版信息如 "In: Proceedings of X, 2020, pp. 1-10" 译成中文描述。

【禁止】
- 不要输出「以下是翻译」「翻译如下」之类前言后记。
- 不要总结、删节、合并段落或添加原文没有的解释。
- 不要输出 markdown 代码围栏包裹整个回答。
- 段落数与原文一一对应：原文有几段，你就输出几段。"""


# ---------------------------------------------------------------- chunking
IMG_LINE = re.compile(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$")
IMG_TAG = re.compile(r"^\s*<img[^>]*>\s*$", re.I)
FENCE_BLOCK = re.compile(r"```[\s\S]*?```")
TABLE_BLOCK = re.compile(r"<table[\s\S]*?</table>", re.I)
FORMULA_BLOCK = re.compile(r"\$\$[\s\S]*?\$\$")
PROTECT_RE = re.compile(
    r"```[\s\S]*?```"          # fenced code
    r"|<table[\s\S]*?</table>"  # html table
    r"|\$\$[\s\S]*?\$\$",      # block formula
    re.I,
)


def _hoist_images(atom: str) -> tuple[list[str], str, list[str]]:
    """把原子块里「独占一行」的图片引用抽出来，返回 (前置图, 文字, 后置图)。

    MinerU 两种常见形态都能覆盖：
      ![](a.jpg)                     → 纯图
      ![](a.jpg)  \\n Figure 10. …    → 图 + 紧随的题注
    """
    before: list[str] = []
    after: list[str] = []
    text: list[str] = []
    seen_text = False
    for ln in atom.split("\n"):
        if not ln.strip():
            continue
        if IMG_LINE.match(ln) or IMG_TAG.match(ln):
            (after if seen_text else before).append(ln)
        else:
            seen_text = True
            text.append(ln)
    return before, "\n".join(text), after


def _split_long_paragraph(p: str, max_chars: int) -> list[str]:
    """超长段落按行切；单行仍超长则按字符硬切。"""
    if len(p) <= max_chars:
        return [p]
    lines = p.split("\n")
    out, cur, cur_len = [], [], 0
    for line in lines:
        if cur_len + len(line) + 1 > max_chars and cur:
            out.append("\n".join(cur))
            cur, cur_len = [line], len(line)
        else:
            cur.append(line)
            cur_len += len(line) + 1
    if cur:
        out.append("\n".join(cur))
    # 兜底：仍有超长单行
    final = []
    for seg in out:
        while len(seg) > max_chars:
            final.append(seg[:max_chars])
            seg = seg[max_chars:]
        if seg:
            final.append(seg)
    return final


def chunk_markdown(md: str, max_chars: int = 3000) -> list[dict]:
    """把 Markdown 切成块，每块 = {"images": [...], "text": str}。

    关键设计：**图片行与 <table> 完全不进模型**。
    实测把图片交给模型时，附录里成串的无题注图片会被整段丢弃
    （MotionAGFormer 丢 6 张）；整表偶尔也会被吃掉（2505.01718 丢 1 张）。
    所以：
      - 独占一行的图片 → 抽成 images，随译文原样拼回，永不丢失；
      - <table> → 抽成 tables，随译文原样拼回（表头保持英文，数据完整）。
    图题/表题（"Figure 10. ..." 这类散文行）仍在 text 里，照常翻译。
    带图的原子块单独成块，保证图与题注的相对位置不被大段文字挤开。
    """
    atoms: list[str] = []
    pos = 0

    def flush_text(text: str) -> None:
        if not text.strip():
            return
        for para in text.split("\n\n"):
            if para.strip():
                atoms.extend(_split_long_paragraph(para, max_chars))

    for m in PROTECT_RE.finditer(md):
        flush_text(md[pos:m.start()])
        atoms.append(m.group(0))
        pos = m.end()
    flush_text(md[pos:])

    chunks: list[dict] = []
    cur: list[str] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if cur:
            chunks.append({"images": [], "text": "\n\n".join(cur)})
            cur, cur_len = [], 0

    for a in atoms:
        if TABLE_BLOCK.fullmatch(a.strip()):
            flush()
            chunks.append({"images": [a], "text": ""})
            continue
        before, text, after = _hoist_images(a)
        if before or after:
            # 带图的原子单独成块：图片贴着自己的题注，不被邻近长段落挤远
            flush()
            if text:
                chunks.append({"images": before + after, "text": text})
            elif before or after:
                chunks.append({"images": before + after, "text": ""})
            continue
        a_len = len(a) + 2
        if cur_len + a_len > max_chars and cur:
            flush()
        cur.append(a)
        cur_len += a_len
    flush()
    return chunks


# ---------------------------------------------------------------- translation
def zh_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    zh = sum(1 for c in letters if "\u4e00" <= c <= "\u9fff")
    return zh / len(letters)


def looks_untranslated(text: str, threshold: float = 0.05) -> bool:
    """字母里中文占比过低 => 模型基本没翻译（回显英文）。"""
    return zh_ratio(text) < threshold


def translate_chunk(client: OpenAI, chunk: str, max_retries: int = 4) -> str:
    if not chunk.strip():
        return ""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": chunk},
                ],
                temperature=0.2,
                stream=False,
            )
            content = resp.choices[0].message.content or ""
            content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
            # 模型偶尔会给整篇套一层 ``` 围栏，剥掉
            if content.startswith("```") and content.rstrip().endswith("```"):
                content = re.sub(r"^```[a-zA-Z]*\n", "", content)
                content = re.sub(r"\n?```$", "", content)
            if not content:
                raise ValueError("empty response")
            # 极短碎片（图题、纯公式、纯表格骨架）没有足够英文可判，不做校验
            if len(re.findall(r"[A-Za-z]{4,}", chunk)) < 6:
                return content
            if looks_untranslated(content):
                raise ValueError(
                    f"response not translated (zh_ratio={zh_ratio(content):.3f})")
            return content
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = min(2 ** attempt * 2, 20)
            time.sleep(wait)
    raise RuntimeError(f"chunk failed after {max_retries} tries: {last_err}")


# ---------------------------------------------------------------- cache
def cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w一-鿿-]+", "_", key)[:120]
    return CACHE_DIR / f"{safe}.json"


def load_cache(key: str, n_chunks: int) -> list[str | None]:
    p = cache_path(key)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("n") == n_chunks:
                return data.get("results")
        except (json.JSONDecodeError, OSError):
            pass
    return [None] * n_chunks


def save_cache(key: str, n_chunks: int, results: list[str | None]) -> None:
    cache_path(key).write_text(
        json.dumps({"n": n_chunks, "results": results}, ensure_ascii=False),
        encoding="utf-8")


# ---------------------------------------------------------------- per paper
def find_new_extract(name: str) -> Path | None:
    """新抽取的英文 md：run_extract_missing.sh 用 <名字中 / 与空格换 _> 作 slug。"""
    slug = name.replace("/", "_").replace(" ", "_")
    root = _manifest.NEW_EXTRACT_DIR
    cands = [p for p in root.glob(f"{slug}*/**/*.md")
             if p.name != "full.md" and p.stat().st_size > 0]
    if not cands:
        return None
    # full.md 与规范化后的 <stem>.md 同时存在时取后者
    cands.sort(key=lambda p: (p.name.endswith("full.md"), -p.stat().st_size))
    return cands[0]


def sync_images(en_md: Path, paper_dir: Path) -> int:
    """把英文抽取旁边的 images/ 同步到论文目录，保证 ![](images/x.jpg) 能解析。"""
    src = en_md.parent / "images"
    if not src.is_dir():
        return 0
    dst = paper_dir / "images"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for img in src.iterdir():
        if img.is_file() and not (dst / img.name).exists():
            (dst / img.name).write_bytes(img.read_bytes())
            n += 1
    return n


def process_paper(row: dict, client: OpenAI, workers: int,
                  chunk_size: int, dry_run: bool) -> dict:
    name = row["name"]
    en_md = row["en_md"]
    pdf = row["pdf"]
    paper_dir = row["paper_dir"]

    if en_md is None or not Path(en_md).exists():
        en_md = find_new_extract(name)
    if en_md is None:
        return {"name": name, "status": "SKIP_NO_EN_MD", "pdf": str(pdf)}

    md = Path(en_md).read_text(encoding="utf-8")
    chunks = chunk_markdown(md, chunk_size)
    # v2：分块器改为图片/表格不进模型，块结构变了，缓存必须换命名空间
    key = f"v2::{name}"
    results = load_cache(key, len(chunks))
    todo = [i for i, r in enumerate(results) if r is None]

    n_img_refs = sum(len(c["images"]) for c in chunks)
    n_tbl_refs = sum(1 for c in chunks
                     if c["images"] and not c["text"]
                     and TABLE_BLOCK.fullmatch(c["images"][0].strip()))

    if dry_run:
        return {"name": name, "status": "PLAN", "chunks": len(chunks),
                "todo": len(todo), "bytes": len(md), "pages": row["pages"],
                "img_atoms": n_img_refs, "tbl_atoms": n_tbl_refs}

    done_before = len(todo)
    if todo:
        t0 = time.time()
        need = [i for i in todo if chunks[i]["text"].strip()]
        skipped = len(todo) - len(need)
        if skipped:
            print(f"      跳过 {skipped} 个纯图/表块（不调 API）", flush=True)
        for i in todo:
            if not chunks[i]["text"].strip():
                results[i] = ""
        if need:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(translate_chunk, client, chunks[i]["text"]): i
                        for i in need}
                for n, f in enumerate(as_completed(futs), 1):
                    i = futs[f]
                    results[i] = f.result()
                    if n % 5 == 0 or n == len(need):
                        print(f"      [{n}/{len(need)}] chunks", flush=True)
                    save_cache(key, len(chunks), results)
        save_cache(key, len(chunks), results)
        elapsed = time.time() - t0
    else:
        elapsed = 0.0

    # 图片与表格按原样拼回，保证零丢失
    assembled: list[str] = []
    for c, r in zip(chunks, results):
        parts = list(c["images"])
        if r:
            parts.append(r)
        if parts:
            assembled.append("\n\n".join(parts))

    out = paper_dir / f"{name}_全文翻译.md"
    out.write_text("\n\n".join(assembled), encoding="utf-8")
    n_img = sync_images(Path(en_md), paper_dir)

    full = out.read_text(encoding="utf-8")
    return {
        "name": name, "status": "OK", "pages": row["pages"],
        "chunks": len(chunks), "translated_this_run": done_before,
        "zh_ratio": round(zh_ratio(full), 3),
        "chars": len(full), "out": str(out), "images_copied": n_img,
        "img_atoms": n_img_refs, "tbl_atoms": n_tbl_refs,
        "seconds": round(elapsed, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只跑名字包含该子串的论文")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk-size", type=int, default=3000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [r for r in _manifest.rows()]
    if args.only:
        rows = [r for r in rows if args.only in r["name"]]
    if args.limit:
        rows = rows[: args.limit]

    if args.dry_run:
        tot = 0
        for r in rows:
            res = process_paper(r, None, 0, args.chunk_size, True)
            if res.get("status") == "PLAN":
                tot += res["chunks"]
                print(f"{res['pages']:>3}p  {res['chunks']:>4} chunks "
                      f"({res['todo']} 待译)  img={res.get('img_atoms',0):>3} "
                      f"tbl={res.get('tbl_atoms',0):>3}  {res['name'][:40]}")
        print(f"\n合计 chunks: {tot}")
        return 0

    import pdf_to_zh_md  # 复用凭据解析（.secrets > 环境变量）
    key = pdf_to_zh_md.load_api_key()
    if not key:
        sys.exit("no API key")
    client = OpenAI(api_key=key, base_url=BASE_URL, timeout=180, max_retries=0)

    report = []
    t_start = time.time()
    for i, r in enumerate(rows, 1):
        print(f"\n[{i}/{len(rows)}] {r['name']}", flush=True)
        try:
            res = process_paper(r, client, args.workers, args.chunk_size, False)
        except Exception as e:  # noqa: BLE001
            res = {"name": r["name"], "status": "FAIL", "error": str(e)[:300]}
        report.append(res)
        if res["status"] == "OK":
            print(f"    OK  zh={res['zh_ratio']}  {res['chunks']} chunks "
                  f"(本轮译 {res['translated_this_run']})  {res['seconds']}s  "
                  f"img+{res['images_copied']}", flush=True)
        else:
            print(f"    {res['status']}: {res.get('error', '')}", flush=True)
        Path("/tmp/translate_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n总耗时 {time.time() - t_start:.0f}s")
    ok = sum(1 for r in report if r["status"] == "OK")
    print(f"成功 {ok}/{len(report)}")
    return 0 if ok == len(report) else 1


if __name__ == "__main__":
    sys.exit(main())
