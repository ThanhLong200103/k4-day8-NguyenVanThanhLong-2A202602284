# Vì sao chọn lô này?

## Năm frame ưu tiên nếu chỉ có ngân sách rà năm ảnh

Nguồn: 50 dòng đầu `outputs/selection_round1.csv` (điểm do model cold start tính, `min_gap_s = 2.0`).

| thứ tự rà | frame | rank CSV | score | t (giây) | U | A | n_boxes / n_ambiguous | lý do |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | frame_0182.jpg | 1 | 0.9591 | 72.8 | 0.918 | 1.000 | 28 / 18 | Điểm cao nhất, nhiều box mập mờ nhất pool (A = 1.0); nằm ở đoạn giữa video, xa các frame khác trong danh sách |
| 2 | frame_0369.jpg | 2 | 0.9324 | 147.6 | 0.932 | 0.889 | 43 / 16 | Cảnh đông (43 box dự đoán), U cao; đại diện cho cụm cuối video |
| 3 | frame_0326.jpg | 4 | 0.9155 | 130.4 | 0.931 | 0.833 | 39 / 15 | Đại diện cho cụm 124–133 s; dòng xe dày, đèn pha chói trên contact sheet |
| 4 | frame_0099.jpg | 8 | 0.9063 | 39.6 | 0.946 | 0.778 | 29 / 14 | U cao nhất trong top 10; là frame duy nhất ở đoạn đầu video (< 45 s) trong top 10, thêm đa dạng thời gian |
| 5 | frame_0380.jpg | 3 | 0.9170 | 152.0 | 0.934 | 0.833 | 40 / 15 | Điểm cao thứ 3; cách frame_0369 4.4 s nên xe đã đổi phần lớn |

**Quyết định về ảnh gần trùng:** tôi **bỏ frame_0331.jpg** (rank 5, score 0.9154, t = 132.4 s) dù điểm
cao hơn frame_0099. Nó chỉ cách frame_0326 đúng 2.0 s. Công cụ vẫn chọn vì đúng bằng `MIN_GAP_S`,
nhưng hai ảnh cùng cụm 130–133 s có chung nhiều xe. Với ngân sách chỉ 5 ảnh, tôi đổi nó lấy
frame_0099 ở đoạn thời gian khác. Tương tự, bỏ frame_0312.jpg (rank 7, t = 124.8 s) vì cùng cụm với
frame_0326.

**Trường hợp model không dự đoán được box:** cột `empty` bằng `False` ở cả 50 dòng đầu, nên
vòng này không có frame trống cần ưu tiên. Frame trống xuất hiện ở vòng sau:
`outputs/selection_round2.csv` có 8 frame `empty = True` (ví dụ frame_0009, frame_0019). Xem mục 5 của REPORT.

## Ba frame thuộc lô 12 ảnh model chọn và bằng chứng

- **frame_0182.jpg**: rank 1, score 0.9591, A = 1.0 (18 box mập mờ, nhiều nhất pool). Sau khi rà,
  `outputs/round1_diff.md` ghi 13 box đề xuất, còn 21 box sau khi sửa: 12 accepted, 1 deleted, 9 added. Điểm
  bất định cao đi cùng việc model bỏ sót nhiều xe thật.
- **frame_0369.jpg**: rank 2, score 0.9324, 43 box dự đoán ở conf ≥ 0.05. Trên contact sheet
  `selection_round1.jpg`, ảnh này có dòng xe dày, nhiều xe đèn pha chói ở làn gần. Diff: 14 box đề xuất,
  27 box sau sửa (1 edited, 1 deleted, **14 added**), là frame phải thêm nhiều box nhất lô. Một xe nhòe ở mép dưới phải
  chỉ được model khoanh phần đầu (ghi trong `REVIEW_LOG.csv`).
- **frame_0331.jpg**: rank 5, score 0.9154, 47 box dự đoán, A = 1.0. Diff: 20 box đề xuất, **5 deleted**
  (nhiều FP nhất lô), 9 added. Model vẽ 2 box chồng nhau lệch khỏi một xe con bên trái (`REVIEW_LOG.csv`).

## Một frame điểm cao nhưng không chọn

**frame_0372.jpg**: rank 6, score 0.9101, cao hơn 7 frame trong lô. Nó không được chọn vì t = 148.8 s, chỉ cách
frame_0369 (t = 147.6 s) 1.2 s, dưới `MIN_GAP_S = 2.0`. Tôi đồng ý bỏ: camera cố định, hai ảnh cách
nhau 1.2 s gần như cùng một tập xe, gán cả hai gần như gấp đôi công mà model học thêm rất ít. Lý do
tương tự với frame_0368 (rank 9, cách 0.4 s) và frame_0330 (rank 12, cách frame_0331 0.4 s).

Nhìn lại thì frame_0331, ảnh mà tôi định bỏ khi chỉ có 5 ảnh, lại cho nhiều box FP nhất (5 deleted).
Nghĩa là ảnh gần trùng về thời gian vẫn có thể chứa lỗi khác loại. Quyết định bỏ nó là đánh đổi chi
phí, không phải vì nó chắc chắn vô ích.

## Điều phép chọn này chưa chứng minh

Điểm U/A chỉ đo **model cold start phân vân** (conf gần 0.5), không đo model **sai** so với nhãn đúng,
và không chứng minh rằng gán nhãn ảnh đó sẽ tăng AP50. Bằng chứng: sau khi train trên 12 ảnh điểm cao
này, AP50 trên test **giảm** từ 0.771 xuống 0.382 (`reports/rounds_table.md`). Chưa có vòng đối chứng
`strategy="random"` với cùng số ảnh, nên cũng không kết luận được chọn theo uncertainty tốt hơn chọn ngẫu nhiên.
