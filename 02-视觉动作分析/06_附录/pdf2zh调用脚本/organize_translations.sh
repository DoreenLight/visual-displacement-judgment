#!/usr/bin/env bash
# ==============================================================================
# organize_translations.sh — 把 pdf2zh 暂存输出归位到论文子目录
# ==============================================================================
# 设计：零硬编码。不再维护 stem→子目录映射表，而是遍历每个论文子目录内的
#       PDF，自动推导 stem，再到暂存区按 stem 找对应产物。
#       新增论文无需修改本脚本。
#
# 职责：
#   1. 把暂存区的 <stem>_zh.md 复制到论文子目录（已存在则 SKIP，保持幂等）
#   2. 把暂存区 images/ 以「硬链接」汇入论文子目录的 images/
#      （硬链接额外磁盘占用≈0；单一软链无法服务同目录多篇论文的不同图集，
#        故采用内容哈希命名下的共享目录 + 硬链接，冲突时自动跳过并告警）
#
# 用法：bash 06_附录/pdf2zh调用脚本/organize_translations.sh
# 兼容：macOS 自带 bash 3.2（不使用 associative array / mapfile）
# ==============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 论文根目录：每个含 PDF 的目录即论文子目录
PAPER_ROOTS="01_参考论文 09_章隆彬_合作团队"

[ -t 1 ] && C_OK=$'\033[32m' && C_SKIP=$'\033[33m' && C_ERR=$'\033[31m' && C_OFF=$'\033[0m' \
         || C_OK="" && C_SKIP="" && C_ERR="" && C_OFF=""

copied=0; skipped=0; imgs=0; miss=0

echo "=============================================================================="
echo "翻译输出归位（自动发现模式）  root=${PROJECT_ROOT}"
echo "=============================================================================="

for pr in $PAPER_ROOTS; do
    base="${PROJECT_ROOT}/${pr}"
    [ -d "$base" ] || continue
    stage_root="${base}/_translation_outputs"
    [ -d "$stage_root" ] || continue

    # 遍历该论文根下的 PDF（跳过暂存区自身）
    find "$base" -name '*.pdf' -type f -not -path "${stage_root}/*" 2>/dev/null | sort | while read -r pdf; do
        stem="$(basename "$pdf" .pdf)"
        subdir="$(dirname "$pdf")"

        # 暂存区产物路径：<stage>/<stem>/<stem>/auto/
        auto_dir="${stage_root}/${stem}/${stem}/auto"
        src_zh="${auto_dir}/${stem}_zh.md"
        dst_zh="${subdir}/${stem}_zh.md"

        # --- 1. 归位中文 Markdown ---
        if [ -f "$src_zh" ]; then
            if [ -f "$dst_zh" ]; then
                echo "  ${C_SKIP}[SKIP]${C_OFF} ${stem}_zh.md 已存在"
            else
                cp "$src_zh" "$dst_zh" && echo "  ${C_OK}[COPY]${C_OFF} ${stem}_zh.md -> ${subdir#"${PROJECT_ROOT}/"}"
            fi
        elif [ -f "$dst_zh" ]; then
            echo "  ${C_SKIP}[SKIP]${C_OFF} ${stem} 暂存区无副本，论文子目录已有权威版本"
        else
            echo "  ${C_ERR}[MISS]${C_OFF} ${stem}: 暂存区与论文子目录均无 _zh.md"
        fi

        # --- 2. 汇入图片（硬链接，额外占用≈0）---
        src_img="${auto_dir}/images"
        [ -d "$src_img" ] || continue
        dst_img="${subdir}/images"
        mkdir -p "$dst_img"
        n=0
        for f in "$src_img"/*; do
            [ -f "$f" ] || continue
            b="$(basename "$f")"
            [ -e "${dst_img}/${b}" ] && continue
            ln "$f" "${dst_img}/${b}" 2>/dev/null || cp "$f" "${dst_img}/${b}"
            n=$((n+1))
        done
        [ "$n" -gt 0 ] && echo "  ${C_OK}[IMG ]${C_OFF} ${stem} 新增 $n 张 -> ${subdir#"${PROJECT_ROOT}/"}/images/"
    done
done

echo "=============================================================================="
echo "完成。请运行 bash 06_附录/自检脚本/verify_kb.sh 复核"
echo "=============================================================================="
