"""
Khôi phục báo cáo RAGAS từ LangSmith traces.

Bối cảnh: lần chạy 03_ragas_evaluation.py, 31/200 job của V2 lỗi (402 — hết credit
OpenRouter). Phiên bản cũ của run_ragas_eval() chỉ lọc None nên 1 giá trị NaN làm
trung bình của V2 thành NaN, dù 169 job vẫn thành công.

Vì LANGCHAIN_TRACING_V2=true, RAGAS đã ghi điểm của TỪNG sample lên LangSmith
(root run "ragas evaluation", outputs["scores"]). Script này đọc lại các run đó,
tính trung bình bỏ qua sample lỗi (NaN/None) — đúng như logic đã sửa trong
03_ragas_evaluation.py — rồi ghi lại data/ragas_report.json.

Cách dùng:
    python recover_ragas_report.py            # lấy 2 run "ragas evaluation" mới nhất
"""
import sys
import json
import math
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import config
from langsmith import Client

METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def mean_valid(values: list) -> tuple:
    """Trung bình bỏ qua None/NaN. Trả về (mean, số sample hợp lệ)."""
    valid = [v for v in values if v is not None and not math.isnan(v)]
    return (sum(valid) / len(valid) if valid else float("nan")), len(valid)


def main():
    client = Client(api_key=config.LANGSMITH_API_KEY)
    runs = list(client.list_runs(
        project_name=config.LANGSMITH_PROJECT, is_root=True, filter='eq(name, "ragas evaluation")',
    ))
    # 03_ragas_evaluation.py đánh giá V1 trước rồi V2 → sắp xếp theo thời gian
    runs = sorted(runs, key=lambda r: r.start_time)[-2:]
    if len(runs) < 2:
        print("⚠️  Không tìm thấy đủ 2 run 'ragas evaluation' trên LangSmith.")
        sys.exit(1)

    report = {}
    for version, run in zip(["v1", "v2"], runs):
        rows = run.outputs["scores"]
        scores = {}
        print(f"\n📊 Prompt {version.upper()} — run {run.id} ({run.start_time:%H:%M:%S} UTC), {len(rows)} samples")
        for m in METRICS:
            scores[m], n_valid = mean_valid([row.get(m) for row in rows])
            print(f"  {m:20s}: {scores[m]:.4f}   ({n_valid}/{len(rows)} sample hợp lệ)")
        report[f"prompt_{version}_scores"] = scores

    best_faith = max(report["prompt_v1_scores"]["faithfulness"], report["prompt_v2_scores"]["faithfulness"])
    report["target_met"] = best_faith >= 0.8

    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for m in METRICS:
        s1, s2 = report["prompt_v1_scores"][m], report["prompt_v2_scores"][m]
        winner = "= hòa" if math.isclose(s1, s2) else ("← V1" if s1 > s2 else "← V2")
        print(f"  {m:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")
    print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8" if report["target_met"]
          else f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8)")

    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main()
