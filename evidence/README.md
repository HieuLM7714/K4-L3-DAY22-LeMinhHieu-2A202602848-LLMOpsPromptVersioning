# Evidence — Day 22: LangSmith + Prompt Versioning

**Học viên:** Lê Minh Hiếu — 2A202602848
**Provider:** OpenRouter (`openai/gpt-4o-mini`, embeddings `openai/text-embedding-3-small`)
**LangSmith project:** `day22-lab`

## Danh sách tệp

| Tệp | Nội dung |
|---|---|
| `01_langsmith_traces.png` | 72 traces `rag-query` (Bước 1, ≥ 50) |
| `01_langsmith_total_traces.png` | Toàn bộ project: 201 traces (≥ 100) |
| `02_prompt_hub.png` | 2 prompt `le-minh-hieu-rag-prompt-v1` / `-v2` trên Prompt Hub |
| `02_ab_routing_log.txt` | Log A/B routing: pull từ Hub, nhãn `[prompt-v1]`/`[prompt-v2]`, V1=19 / V2=31 |
| `03_ragas_scores.png` | Bảng so sánh V1 vs V2 |
| `03_ragas_report.json` | Bản sao `data/ragas_report.json` |
| `03_ragas_recovery_log.txt` | Log khôi phục điểm V2 từ LangSmith (xem ghi chú bên dưới) |
| `04_pii_demo_log.txt` | 6 test case PII (email, phone, SSN, thẻ tín dụng, multi-PII, sạch) |
| `04_json_demo_log.txt` | 5 test case JSON (hợp lệ, fences, nháy đơn, dấu phẩy thừa, không sửa được) |

> `04_pii_demo_log.txt` và `04_json_demo_log.txt` cùng nội dung vì `04_guardrails_validator.py` in cả 2 demo trong 1 lần chạy (đúng hướng dẫn CHECKPOINTS).

## Kết quả RAGAS

| Metric | V1 (ngắn gọn) | V2 (có cấu trúc) | Cao hơn |
|---|---|---|---|
| faithfulness | 0.9605 | **0.9625** | V2 |
| answer_relevancy | **0.9116** | 0.8926 | V1 |
| context_recall | 1.0000 | 1.0000 | hòa |
| context_precision | 0.9383 | **0.9437** | V2 |

Faithfulness ≥ 0.9 ở **cả 2** phiên bản.

## Phân tích V1 vs V2

**Hai prompt khác nhau ở đâu:**
- **V1** — "trợ lý thân thiện", trả lời **2–4 câu**, nói thẳng "không biết" nếu context không có.
- **V2** — "chuyên gia phân tích", yêu cầu **xác định facts trong context trước** rồi viết **3–5 câu** có tổ chức, "không suy đoán ngoài context".

**V2 nhỉnh hơn về faithfulness (0.9625 vs 0.9605).** V2 bắt model trích facts từ context trước khi viết và cấm suy đoán, nên các claim trong câu trả lời bám sát tài liệu hơn. Tuy vậy cả 2 đều rất cao vì cả 2 prompt đều ràng buộc "chỉ dựa trên context" — đây là yếu tố quyết định faithfulness, còn khác biệt về giọng văn chỉ ảnh hưởng nhỏ.

**V1 cao hơn rõ về answer_relevancy (0.9116 vs 0.8926).** RAGAS tính chỉ số này bằng cách sinh ngược câu hỏi từ câu trả lời rồi so embedding với câu hỏi gốc. Câu trả lời 2–4 câu của V1 đi thẳng vào câu hỏi; câu trả lời 3–5 câu của V2 thường thêm bối cảnh/chi tiết phụ, làm câu hỏi sinh ngược "rộng" hơn câu hỏi gốc → điểm thấp hơn. Đây là trade-off điển hình: **chi tiết hơn ≠ liên quan hơn**.

**context_recall / context_precision gần như bằng nhau.** Hai chỉ số này đo chất lượng **retriever** (cùng FAISS index, cùng `k=3`, cùng câu hỏi), không phụ thuộc prompt. Chênh lệch nhỏ ở context_precision (0.9383 vs 0.9437) đến từ nhiễu của LLM chấm điểm và việc V2 được tính trên ít sample hơn (xem ghi chú), không phải do prompt.

**Kết luận:** chênh lệch giữa 2 prompt đều ≤ 0.02, nhỏ so với nhiễu đánh giá trên 50 mẫu. Nếu ưu tiên câu trả lời đúng trọng tâm (chatbot hỏi đáp) → chọn **V1**; nếu ưu tiên bám sát tài liệu, giải thích đầy đủ hơn → **V2**. Muốn kết luận chắc chắn cần nhiều mẫu hơn hoặc chạy lặp nhiều lần.

## Ghi chú về điểm V2 (minh bạch)

Trong lần chạy `03_ragas_evaluation.py`, tài khoản OpenRouter hết credit gần cuối phần chấm V2: **31/200 job** (50 sample × 4 metric) trả về lỗi `402`. Phiên bản cũ của `run_ragas_eval()` chỉ lọc `None` mà không lọc `NaN`, nên trung bình V2 bị in ra `nan` dù 169 job vẫn thành công.

Vì tracing LangSmith đang bật, RAGAS đã ghi **điểm từng sample** của cả 2 lần chấm lên LangSmith (root run `ragas evaluation`). `src/recover_ragas_report.py` đọc lại 2 run đó và tính trung bình bỏ qua sample lỗi:
- V1 tính lại khớp **chính xác** với kết quả in ra khi chạy (0.9605 / 0.9116 / 1.0000 / 0.9383) → phương pháp đúng.
- V2 được tính trên số sample hợp lệ: faithfulness 42/50, answer_relevancy 43/50, context_recall 44/50, context_precision 40/50.

`03_ragas_evaluation.py` đã được sửa để bỏ qua NaN và in số sample bị bỏ qua (cùng logic với script khôi phục).
