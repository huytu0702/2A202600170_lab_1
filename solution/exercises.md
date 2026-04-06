# Ngày 1 — Bài Tập & Phản Ánh

## Nền Tảng LLM API | Phiếu Thực Hành

**Thời lượng:** 1:30 giờ  
**Cấu trúc:** Lập trình cốt lõi (60 phút) → Bài tập mở rộng (30 phút)

---

## Phần 1 — Lập Trình Cốt Lõi (0:00–1:00)

Chạy các ví dụ trong Google Colab tại: [https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing](https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing)  

Triển khai tất cả TODO trong `template.py`. Chạy `pytest tests/` để kiểm tra tiến độ.

**Điểm kiểm tra:** Sau khi hoàn thành 4 nhiệm vụ, chạy:

```bash
python template.py
```

Bạn sẽ thấy output so sánh phản hồi của GPT-4o và GPT-4o-mini.

---

## Phần 2 — Bài Tập Mở Rộng (1:00–1:30)

### Bài tập 2.1 — Độ Nhạy Của Temperature

Gọi `call_openai` với các giá trị temperature 0.0, 0.5, 1.0 và 1.5 sử dụng prompt **"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)

> Ở temperature 0.0, mô hình luôn trả về cùng một câu trả lời rất ngắn gọn và mang tính sự thật rõ ràng (ví dụ: số liệu dân số, tên địa danh), gần như không có sự thay đổi giữa các lần gọi. Khi tăng lên 0.5 và 1.0, câu trả lời trở nên đa dạng hơn về cách diễn đạt, có thể thêm ngữ cảnh hay chi tiết thú vị hơn. Ở 1.5, phản hồi đôi khi trở nên sáng tạo hoặc bất thường, có thể pha trộn nhiều chi tiết khác nhau hoặc diễn đạt theo kiểu ít phổ biến hơn.

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**

> Em sẽ đặt temperature khoảng **0.1–0.2**. Chatbot hỗ trợ khách hàng cần đưa ra câu trả lời nhất quán, chính xác và đáng tin cậy — ví dụ: thông tin về chính sách hoàn trả, trạng thái đơn hàng, hay hướng dẫn kỹ thuật. Nếu temperature quá cao, mô hình có thể tạo ra thông tin sai lệch hoặc mâu thuẫn, gây mất tin cậy với khách hàng.

---

### Bài tập 2.2 — Đánh Đổi Chi Phí

Xem xét kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người thực hiện 3 lần gọi API, mỗi lần trung bình ~350 token.

**Ước tính xem GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này:**

> Gọi `x` là số input tokens và `y` là số output tokens trong mỗi lần gọi API.
>
> Giá của từng model là:
>
> - **GPT-4o**: input = **$2.50 / 1M tokens**, output = **$10.00 / 1M tokens**
> - **GPT-4o-mini**: input = **$0.15 / 1M tokens**, output = **$0.60 / 1M tokens**
>
> Khi đó:
>
> `Cost_4o = x·2.50 + y·10.00`
>
> `Cost_4o-mini = x·0.15 + y·0.60`
>
> Ta có:
>
> `2.50 ~ 16.67 × 0.15`
>
> `10.00 ~ 16.67 × 0.60`
>
> nên:
>
> `Cost_4o = x·2.50 + y·10.00`
>
>         `~ 16.67·(x·0.15 + y·0.60)`
>
>         `~ 16.67·Cost_4o-mini`
>
> Vậy:
>
> `Cost_4o ~ 16.67 · Cost_4o-mini`
>
> Suy ra GPT-4o đắt hơn khoảng **16.67 lần** so với GPT-4o-mini nếu cùng số lượng input và output tokens trong workload này.

**Mô tả một trường hợp mà chi phí cao hơn của GPT-4o là xứng đáng, và một trường hợp GPT-4o-mini là lựa chọn tốt hơn:**

> **GPT-4o phù hợp hơn:** Phân tích hợp đồng pháp lý hoặc chẩn đoán y tế — nơi độ chính xác, khả năng lập luận phức tạp và chất lượng phản hồi là ưu tiên hàng đầu, sai sót có thể gây hậu quả nghiêm trọng.
>
> **GPT-4o-mini phù hợp hơn:** Phân loại email khách hàng, tóm tắt nội dung ngắn, hay trả lời câu hỏi FAQ — các tác vụ đơn giản, lặp đi lặp lại với khối lượng lớn, khi chất lượng của mini đã đủ tốt và tiết kiệm chi phí là yếu tố quyết định.

---

### Bài tập 2.3 — Trải Nghiệm Người Dùng với Streaming

**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?** (1 đoạn văn)

> Streaming quan trọng nhất khi phản hồi của mô hình dài (ví dụ: sinh code, viết bài luận, giải thích khái niệm phức tạp) và trong các ứng dụng chatbot tương tác thực — người dùng nhận được token đầu tiên ngay lập tức thay vì chờ toàn bộ phản hồi, tạo cảm giác phản hồi nhanh và tự nhiên hơn. Ngược lại, non-streaming phù hợp hơn trong các pipeline tự động (API-to-API), khi cần toàn bộ phản hồi trước khi xử lý tiếp (ví dụ: phân tích JSON, phân loại, tóm tắt để lưu vào database), hoặc trong batch processing — khi trải nghiệm người dùng không quan trọng bằng tính đơn giản và ổn định của luồng xử lý.

## Danh Sách Kiểm Tra Nộp Bài

- Tất cả tests pass: `pytest tests/ -v`
- `call_openai` đã triển khai và kiểm thử
- `call_openai_mini` đã triển khai và kiểm thử
- `compare_models` đã triển khai và kiểm thử
- `streaming_chatbot` đã triển khai và kiểm thử
- `retry_with_backoff` đã triển khai và kiểm thử
- `batch_compare` đã triển khai và kiểm thử
- `format_comparison_table` đã triển khai và kiểm thử
- `exercises.md` đã điền đầy đủ
- Sao chép bài làm vào folder `solution` và đặt tên theo quy định

