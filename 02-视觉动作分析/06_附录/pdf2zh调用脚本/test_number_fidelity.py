"""数字保真判据回归测试：必须能抓真错，又不能把单位换算/DOI/日期改写误判成错。

    python test_number_fidelity.py
"""
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "v", Path(__file__).parent / "verify_translations.py")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)

body = V._body_for_numbers
CASES = [
    # (说明, 英文, 译文, 期望判为「丢失」的数字集合)
    ("中英数量级换算", "affecting 2.4 billion people", "约有 24 亿人患病", set()),
    ("万/亿换算", "13.7 million people", "1370 万人", set()),
    ("万换算正确", "around 1.3 million pixels", "约 130 万像素", set()),
    ("真错 8.57→8.47", "average MAE of 8.57", "平均 MAE 为 8.47", {"8.57"}),
    ("真错 8.15→8.51", "MAE of 8.15 and 8.05", "MAE 为 8.51 和 8.05", {"8.15"}),
    ("真错 1.3M→13万", "around 1.3 million pixels", "约 13 万像素", {"1300000"}),
    ("DOI 冒号全角化", "doi: 10.3390/s18041004.", "doi：10.3390/S18041004", set()),
    ("日期改写", "published 13.06.22", "发表于 2022 年 6 月 13 日", set()),
    ("arXiv 带版本号", "arXiv:2404.16752v1", "arXiv 2404.16752", set()),
    ("小数保真", "PCC of 0.969 and 0.900", "PCC 为 0.969 与 0.900", set()),
    ("正文真错要抓", "## Results\n\nThe MAE was 8.57.",
     "## 结果\n\n平均 MAE 为 8.47。", {"8.57"}),
    ("参考文献 DOI 不误报",
     "## Results\n\nMAE 8.57.\n\n## References\n\n[1] doi: 10.1016/j.jbi.2022.103828.",
     "## 结果\n\nMAE 为 8.57。\n\n## 参考文献\n\n[1] doi：10.1016/j.jbi.2022.103828。", set()),
    ("参考文献后的大表不误报",
     "正文 MAE 8.57。\n\n## 参考文献\n\n[1] x.",
     "正文 MAE 为 8.57。\n\n## 参考文献\n\n[1] x。\n\n"
     "<table><tr><td>10.49</td><td>10.59</td></tr></table>", set()),
]

ok = 0
for name, en, zh, expect in CASES:
    got = V._lost_data_nums(body(en), body(zh))
    good = got == expect
    ok += good
    print(("  OK  " if good else "  BAD "), name,
          "" if good else f"got={got} expect={expect}")
print(f"\n通过 {ok}/{len(CASES)}")
sys.exit(0 if ok == len(CASES) else 1)
