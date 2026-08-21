# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Nguyễn Trần Nghĩa |
| MSSV | 2A202601664 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/Revlis15/Track2_Day21_2A202601664_NguyenTranNghia |
| Ngày nộp | 21/08/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |
| 4 | 150 | 0.1 | 4 | 0.7156 | 0.8760 |

**Bộ siêu tham số đã chọn:** `n_estimators=150`, `learning_rate=0.1`, `max_depth=4`.

**Lý do:** Bộ tham số này đạt `f1_score` cao nhất (0.7156) trên tập đánh giá holdout, vượt qua ngưỡng kiểm định chất lượng 0.65. Lần chạy có accuracy cao nhất (Lần 1 với 0.8780) không trùng với lần có f1-score cao nhất, minh chứng rằng accuracy cao có thể bị chi phối bởi lớp đa số thay vì chất lượng thực sự trên lớp thu nhập cao. Giữa `n_estimators` và `learning_rate` có sự đánh đổi: khi giảm learning_rate xuống 0.05 và số cây ít (50 cây), mô hình bị underfitting (F1 chỉ đạt 0.6051). Ngược lại, khi tăng số cây lên 150 với độ sâu 4, thuật toán Gradient Boosting sửa lỗi tuần hoàn tốt hơn và cải thiện F1 rõ rệt.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult có phân bố lớp mất cân bằng đáng kể khi lớp thu nhập cao (>50K) chỉ chiếm khoảng 24.8% tổng số mẫu. Trong tình huống này, một mô hình dự đoán tầm thường luôn gán nhãn "thu nhập thấp" cho mọi đối tượng vẫn đạt được Accuracy lên tới 75.2% nhưng hoàn toàn vô dụng vì F1-score của lớp dương bằng 0.000. 

Chỉ số F1-score trên lớp dương tính toán trung bình điều hòa giữa Precision và Recall, đo lường chính xác năng lực nhận diện nhóm khách hàng thu nhập cao mà không bị thổi phồng bởi lớp đa số. Khi đánh giá, tuyệt đối không dùng `average="weighted"` hay `average="macro"` vì việc tính trung bình sẽ bị kéo lệch bởi lớp chiếm 75% dữ liệu, làm sai lệch quyết định của Quality Gate trong CI/CD.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Không thể tạo file key JSON từ Service Account | Chính sách bảo mật Organization của GCP chặn lệnh tạo Service Account key | Cấp quyền đọc public cho bucket GCS và cấu hình OS Login / SSH key trực tiếp cho VM |
| Lỗi unpickle model trên máy ảo Cloud | Phiên bản scikit-learn mặc định trên VM (1.7.2) không khớp với bản huấn luyện (1.4.2) | Cài đặt chính xác `scikit-learn==1.4.2` theo `requirements.txt` trên môi trường VM |
| Thư viện MLflow thiếu pkg_resources khi tạo bằng uv | uv khởi tạo môi trường Python 3.11 mới không kèm setuptools cũ | Cài đặt thêm gói `setuptools<71` vào môi trường ảo |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7156 | 0.8760 |
| Bước 3 (thêm `train_batch2`) | 0.7248 | 0.8800 |

**Nhận xét:** Khi bổ sung thêm `train_batch2.csv`, số lượng mẫu tăng gấp đôi giúp mô hình học được ranh giới quyết định tổng quát hơn, nâng `f1_score` từ 0.7156 lên 0.7248. Việc pipeline CI/CD tự động kích hoạt và cập nhật mô hình mới sau một lệnh push minh chứng cho tính hiệu quả của quy trình Continuous Training.

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

- [x] Bonus 1 - Tracking MLflow từ xa với DagsHub: Đã kết nối repo với DagsHub và cấu hình tự động log experiments lên server MLflow từ xa tại DagsHub.
- [x] Bonus 2 - Điều chỉnh ngưỡng quyết định: Quét ngưỡng xác suất [0.1 - 0.95] tìm được ngưỡng tối ưu 0.35 giúp tăng F1 từ 0.7156 lên 0.7339.
- [x] Bonus 3 - Báo cáo precision / recall tự động: Tự động tạo Confusion Matrix và Classification Report chi tiết lưu trong `outputs/detail.txt` và GitHub artifact.
- [x] Bonus 4 - Hoàn trả về phiên bản trước / Quality Gate an toàn: Pipeline tự động kiểm định và chặn release (báo lỗi Quality Gate) khi F1 < 0.65 (minh chứng trong ảnh `06-quality-gate-chan.png`).
- [x] Bonus 5 - Cảnh báo lệch lạc dữ liệu: Kiểm tra tỷ lệ lớp dương trong tập huấn luyện (24.77% so với chuẩn 24.80%) và ghi nhận vào `outputs/report.json`.
