# Ngày 1 — Bài Tập & Phản Ánh
## Nền Tảng LLM API | Phiếu Thực Hành

**Thời lượng:** 1:30 giờ  
**Cấu trúc:** Lập trình cốt lõi (60 phút) → Bài tập mở rộng (30 phút)

---

## Phần 1 — Lập Trình Cốt Lõi (0:00–1:00)

Chạy các ví dụ trong Google Colab tại: https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing

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
> *Khi temperature = 0.0, model luôn trả về cùng một câu trả lời xác định, ngắn gọn và bám sát sự thật đã biết. Khi tăng temperature lên 0.5 và 1.0, phản hồi trở nên đa dạng hơn, dùng nhiều từ ngữ phong phú và đôi khi chọn các góc độ khác nhau để trình bày. Ở temperature = 1.5, câu trả lời có thể sáng tạo nhưng đôi khi không mạch lạc hoặc lặp ý, cho thấy model đang "ngẫu nhiên hóa" quá mức.*

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> *Mình sẽ đặt temperature = 0.2 cho chatbot hỗ trợ khách hàng. Ở mức này, model cho ra câu trả lời nhất quán, chính xác và đáng tin cậy — điều quan trọng nhất trong ngữ cảnh hỗ trợ khách hàng nơi thông tin sai lệch có thể gây hại. Một chút biến thiên nhỏ vẫn đủ để câu trả lời không bị cứng nhắc hoàn toàn.*

---

### Bài tập 2.2 — Đánh Đổi Chi Phí
Xem xét kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người thực hiện 3 lần gọi API, mỗi lần trung bình ~350 token.

**Ước tính xem GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này:**
> *Tổng số token mỗi ngày: 10.000 × 3 × 350 = 10.500.000 token (giả sử input/output chia đôi ~175K input, ~175K output mỗi batch).

GPT-4o: (5.25M × $5.00 + 5.25M × $20.00) / 1.000.000 ≈ $131.25/ngày
GPT-4o-mini: (5.25M × $0.15 + 5.25M × $0.60) / 1.000.000 ≈ $3.94/ngày
GPT-4o đắt hơn GPT-4o-mini khoảng 33 lần cho cùng workload này.*

**Mô tả một trường hợp mà chi phí cao hơn của GPT-4o là xứng đáng, và một trường hợp GPT-4o-mini là lựa chọn tốt hơn:**
> *GPT-4o xứng đáng khi xây dựng công cụ phân tích hợp đồng pháp lý hoặc tư vấn y tế, nơi độ chính xác cao và khả năng suy luận phức tạp là bắt buộc — một lỗi nhỏ có thể gây hậu quả nghiêm trọng. Ngược lại, GPT-4o-mini là lựa chọn tốt hơn cho các tác vụ đơn giản như phân loại email, trả lời FAQ, hoặc tóm tắt nội dung ngắn — nơi tốc độ và chi phí thấp quan trọng hơn độ tinh tế của câu trả lời.*

---

### Bài tập 2.3 — Trải Nghiệm Người Dùng với Streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?** (1 đoạn văn)
> *Streaming đặc biệt quan trọng trong các ứng dụng tương tác trực tiếp với người dùng như chatbot, trợ lý ảo, hay công cụ viết lách — nơi người dùng cần thấy phản hồi ngay lập tức thay vì chờ đợi toàn bộ output. Khi response dài (vài trăm token trở lên), streaming giúp giảm cảm giác "chờ đợi" và tạo trải nghiệm tự nhiên giống như đang nói chuyện với người thật. Ngược lại, non-streaming phù hợp hơn cho các tác vụ xử lý hàng loạt (batch processing) chạy ngầm, hoặc khi output cần được xử lý toàn bộ trước khi hiển thị — ví dụ như sinh code rồi chạy linting, hay dịch văn bản rồi định dạng lại — vì trong những trường hợp đó việc nhận từng token riêng lẻ không mang lại lợi ích gì.*


## Danh Sách Kiểm Tra Nộp Bài
- [ ] Tất cả tests pass: `pytest tests/ -v`
- [ ] `call_openai` đã triển khai và kiểm thử
- [ ] `call_openai_mini` đã triển khai và kiểm thử
- [ ] `compare_models` đã triển khai và kiểm thử
- [ ] `streaming_chatbot` đã triển khai và kiểm thử
- [ ] `retry_with_backoff` đã triển khai và kiểm thử
- [ ] `batch_compare` đã triển khai và kiểm thử
- [ ] `format_comparison_table` đã triển khai và kiểm thử
- [ ] `exercises.md` đã điền đầy đủ
- [ ] Sao chép bài làm vào folder `solution` và đặt tên theo quy định 
