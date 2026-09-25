# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Nguyễn Văn Thành Long

Công cụ gán nhãn đã dùng: CVAT (chạy Docker trên máy cá nhân), xuất định dạng Ultralytics YOLO Detection 1.0

## 1. Dữ liệu và cách chia tập

Theo `data/DATA.md`, 400 frame lấy từ **một** cảnh quay liên tục 160 s, camera cố định trên cầu vượt,
2.5 frame/giây. Hai frame liền nhau chỉ cách 0.4 s, gần như giống hệt nhau, và mỗi xe ở lại trong khung
hình vài giây. Vì vậy, dữ liệu được chia theo thời gian: 20 ảnh test ở 4 đoạn quanh giây 20/60/100/140,
112 ảnh vùng đệm bị loại, và 268 ảnh pool. Ảnh pool gần nhất vẫn cách ảnh test 4.4 s.

Nếu chia ngẫu nhiên, một ảnh test gần như chắc chắn có "hàng xóm" cách 0.4 s trong tập train, chứa
cùng những chiếc xe ở gần cùng vị trí. Model sẽ được chấm trên xe nó đã thấy (rò rỉ dữ liệu). Khi đó AP50,
precision và recall đều bị **lệch lên (lạc quan)**: số đo phản ánh khả năng nhớ, không phải khả năng
tổng quát sang đoạn video mới. Vùng đệm 4 s đảm bảo phần lớn xe trong ảnh test đã rời khung hình trước
khi đến đoạn pool gần nhất.

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng vòng 0 từ `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Theo `outputs/metrics_round0.json`: TP 197, FP 16, FN 206 trên 403 box tham chiếu. Model **chính xác nhưng
bỏ sót nhiều**: precision 0.925 nhưng recall chỉ 0.489.

Trong `outputs/compare_round0.jpg`, box vàng (FN) tập trung ở hai nhóm:

- **Xe ở gần, đèn pha rất chói ở nửa dưới ảnh**: frame_0250 bỏ sót xe lớn góc dưới trái và xe bên trái
  làn gần; frame_0350 bỏ sót hai xe gần camera ở dưới trái và dưới phải; frame_0050 bỏ sót xe dưới trái.
  Đèn pha làm cháy sáng thân xe, không giống xe ban ngày trong COCO.
- **Xe ở xa gần đường chân trời, chỉ thấy cụm đèn**: frame_0350 có cụm xe xa bên trái hầu hết là vàng,
  và frame_0150 cũng vậy ở hàng xe trên cùng.

Recall theo kích thước khớp với quan sát: small 0.182 (66 box) thấp hơn nhiều so với medium 0.547 (296 box)
và large 0.561 (41 box). Model COCO gần như không nhận ra xe nhỏ ban đêm. Ngay cả xe lớn cũng chỉ được
phát hiện khoảng một nửa, do bị lóa đèn.

**Ca cần rà lại nhãn tham chiếu:** ở frame_0150, phía phải làn đối diện có một xe đèn hậu đỏ với vệt phản
chiếu đỏ trên mặt đường. Tham chiếu có **ba box chồng lên nhau** ở đúng chỗ đó, và model cold start cũng vẽ
box đỏ (FP) ở vị trí này. Có thể tham chiếu đã tách một xe thành nhiều box, hoặc gán cả vệt đèn phản chiếu
(guideline yêu cầu không gán). Nếu đúng vậy thì FP/FN ở đây là lỗi tham chiếu, không phải lỗi model. Cần
người xem lại ảnh gốc trước khi kết luận. Tương tự, box tham chiếu rất nhỏ ở mép phải frame_0250 chỉ là một
vệt sáng, cần xác nhận có phải xe bị cắt ở mép hay không.

## 3. Chiến lược chọn mẫu

`score = W_U·U + W_A·A + W_D·D` với trọng số 0.5 / 0.3 / 0.2 (`tools/al_select.py`):

- **U (bất định)**: với mỗi box dự đoán, `u = 1 − |2c − 1|`, bằng 1 khi conf = 0.5. U là trung bình 5 giá trị
  u lớn nhất trong frame, đo mức phân vân ở những box khó nhất.
- **A (mập mờ)**: số box có 0.15 ≤ conf < 0.50, chia cho giá trị lớn nhất trong pool. Frame có nhiều box
  lưng chừng thì A cao.
- **D (đa dạng)**: khoảng cách thời gian đến frame đã gán gần nhất, chặn ở 10 s. Vòng 1 chưa có frame nào
  được gán, nên D = 1 cho mọi frame (cột D trong CSV đều bằng 1.0).
- **`MIN_GAP_S = 2.0`**: chọn tham lam theo score, bỏ frame cách một frame đã chọn trong lô dưới 2 s, để không
  tốn công gán hai ảnh gần trùng.

Dẫn chứng từ `reports/SELECTION.md` và CSV:

- frame_0182 (rank 1, A = 1.0, 18 box mập mờ) → diff: 9 added, 1 deleted.
- frame_0369 (rank 2, 43 box dự đoán) → diff: 14 added, nhiều nhất lô. Đây là cảnh đông nên công gán nhãn cao.
- frame_0331 (rank 5) → diff: 5 deleted, nhiều FP nhất lô.
- Frame bị loại vì gần trùng: frame_0372 (rank 6, score 0.9101) cách frame_0369 1.2 s. Tương tự, frame_0368
  và frame_0330 (cách 0.4 s). Nhờ vậy lô 12 ảnh phủ từ 39.6 s đến 156.8 s thay vì dồn vào cụm 130–150 s.

**Điểm bất định không chứng minh ảnh đó sẽ cải thiện model.** U/A chỉ nói model cold start *phân vân*, không
nói model *sai*. Bằng chứng: frame_0099 có U = 0.946, cao nhất top 10, nhưng 13/13 box đề xuất đều được
accepted. Lỗi thật của frame này là 8 xe không có trong pre-label (conf < 0.25) và phải thêm tay. Điểm
cao báo đúng là "có vấn đề", nhưng không chỉ ra được vấn đề nằm ở đâu. Ngoài ra,
sau khi train trên 12 ảnh điểm cao này, AP50 giảm mạnh (mục 4). Muốn biết chiến lược có hiệu quả hay không,
cần so với một vòng `strategy="random"` cùng số ảnh.

## 4. Các vòng học chủ động (active learning)

Bảng từ `reports/rounds_table.md` (test 20 ảnh, 403 box tham chiếu, bỏ qua 14 box cao dưới 16 px, IoU 0.5):

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 263 | 0.382 | -0.390 | 1.000 | 0.007 | 0.015 | 0.000 | 0.010 | 0.000 |

**Vòng 1: mức độ sửa nhãn gợi ý** (`outputs/round1_diff.md`): 12 ảnh, model đề xuất 169 box, sau khi sửa
còn 263 box. Trong đó **154 accepted, 5 edited, 10 deleted (FP của model), 104 added (FN của model)**,
accept rate 91%. Có đến 104/263 box (≈ 40%) phải thêm tay, khớp với recall test 0.489 của cold start:
lỗi chính của pre-label là bỏ sót, không phải vẽ sai.

**AP50**: 0.771 → 0.382, giảm 0.390 so với cold start (đây cũng là vòng trước). Theo `metrics_round1.json`,
chỉ còn TP 3, FP 0, FN 400. Recall ở mọi nhóm kích thước gần như bằng 0 (small 0.000, medium 0.010,
large 0.000). Không nhóm xe nào tốt lên; mọi nhóm đều xấu đi.

**Ca đổi kết quả trong `outputs/compare_round1.jpg`**: frame_0050 từ cold start TP 11 / FP 2 / FN 7 xuống
round 1 TP 1 / FP 0 / FN 17. Frame_0350 từ TP 9 xuống TP 0 / FN 23. Ở cột round 1, gần như mọi box tham chiếu
đều vàng và không có box đỏ nào, tức model **không đưa ra box nào ở conf ≥ 0.25**. AP50 vẫn còn 0.382 (tính
trên mọi ngưỡng conf) cho thấy model vẫn xếp hạng đúng một phần ở conf thấp. Vậy vấn đề là **confidence bị
sụp**, không phải model mất hẳn khả năng định vị.

Nguyên nhân có thể kiểm: notebook train lại từ `yolov8n.pt` với một lớp `car` duy nhất, nên đầu phân loại
phải học lại từ đầu, trong khi chỉ có 12 ảnh với `batch=16`. Tức khoảng 1 bước cập nhật mỗi epoch, 50 bước
cho cả 50 epoch, quá ít để conf vượt 0.25. Cách kiểm tra: chạy predict model vòng 1 với conf 0.01 trên test
và xem phân bố conf, đồng thời xem đường loss và mAP val trong `runs/`. Nhãn đã được nạp đủ: box train = 263
khớp với `round1_diff.md`, nên nguyên nhân không nằm ở việc thiếu file nhãn.

**Phân biệt ba nguồn bằng chứng:**

- *Quan sát độc lập* (`BLIND_SCAN.md`, đã khóa trước khi mở pre-label): frame_0099 ước lượng khoảng 21 xe,
  và dự đoán hai vị trí AI sẽ sót: SUV bị cắt ở mép dưới giữa ảnh, và sedan đèn chói ở làn giữa.
- *Lỗi pre-label đã sửa* (`round1_diff.md`, `REVIEW_LOG.csv`): frame_0099 thực tế có 13 accepted + 8 added
  = 21 box, đúng bằng ước lượng. Cả hai vị trí dự đoán trong blind scan đều nằm trong 8 box added
  (x480–627, y571–719 và x712–813, y412–500).
- *Kết quả mô hình sau train* (`metrics_round1.json`, `compare_round1.jpg`): AP50 giảm do conf sụp. Điều
  này không mâu thuẫn với việc nhãn đã được sửa tốt hơn, vì hai việc được đo ở hai chỗ khác nhau.

**Ca khó theo guideline**: frame_0227 có xe tải đèn pha ở làn gần, đứng sát một xe con. Model vẽ một box trùm
nóc xe tải lẫn xe con bên cạnh, và một box xe tải quá rộng lấn sang xe con. Theo guideline ("hai xe đứng sát
nhau → vẽ hai box riêng"; "box ôm sát thân xe"), tôi xóa box gộp và thu box xe tải ôm sát thân. Một ca khác:
frame_0369 có xe nhòe do chuyển động ở mép dưới phải. Theo guideline ("xe bị nhòe vẫn gán, box ôm vùng
nhòe"), tôi nới box từ x1045–1105 ra x1045–1185. Quy ước tôi giữ cho mọi vòng: không tính vệt đèn pha trên
mặt đường; xe ở xa chỉ còn cụm đèn thì vẫn vẽ box theo thân xe đoán được nếu cao ≥ khoảng 16 px.

## 5. Kết luận và giới hạn

So với cold start, vòng 1 **kém hơn rõ rệt**: AP50 0.771 → 0.382, recall 0.489 → 0.007. Mức giảm này lớn hơn
rất nhiều so với ngưỡng nhiễu 0.01 mà `DATA.md` nêu, nên đây là thay đổi thật, không phải dao động của tập
test nhỏ.

**Tôi dừng, chưa làm vòng 2 theo đúng quy trình hiện tại.** Lý do: model vòng 1 hầu như không cho box nào ở
conf ≥ 0.25, nên điểm chọn ảnh vòng 2 do nó tính ra không đáng tin. Trong `outputs/selection_round2.csv`,
U của các frame đầu chỉ khoảng 0.4–0.5, và có 8 frame `empty = True`. Pre-label vòng 2 cũng gần như trống, ví dụ
`to_label/round2/labels/train/frame_0009.txt` có 0 byte. Việc cần làm trước là sửa cách train: tăng số
epoch hoặc bước cập nhật, hoặc fine-tune từ trọng số đã biết lớp car (COCO), rồi train lại vòng 1 và kiểm
tra AP50 phục hồi. Sau đó mới gán thêm.

Hai ca đề xuất cho vòng sau:

1. **Frame trống kiểu frame_0009 / frame_0019** (`selection_round2.csv`, `empty = True`): camera này luôn có
   xe, nên frame trống là model bỏ sót toàn bộ. Chi phí rà cao: pre-label trống, phải vẽ tay khoảng 20–25 xe
   mỗi ảnh (lô vòng 1 trung bình 263/12 ≈ 22 box/ảnh). Nguy cơ gần trùng: frame_0019, frame_0020, frame_0021
   (7.6 / 8.0 / 8.4 s) cùng trống và gần như là một ảnh, chỉ nên lấy một.
2. **Xe nhỏ ở xa, cảnh đông kiểu frame_0330 / frame_0372** (điểm vòng 1 cao nhưng bị loại vì gần trùng):
   recall small của cold start chỉ 0.182 (66 box). frame_0330 có 53 box dự đoán, nhiều nhất top 50, nên chi
   phí rà rất cao. Hai frame này cách frame đã gán (frame_0331, frame_0369) chỉ 0.4–1.2 s, nên chọn lại gần
   như là gán trùng. Nên chọn một frame đông khác ở cách xa hơn 4 s.

**Giới hạn ảnh hưởng đến kết luận**:

- Tập test chỉ có 20 ảnh, thuộc 4 đoạn thời gian, nên chỉ phát hiện được thay đổi lớn. Chênh dưới khoảng 0.01
  không có ý nghĩa.
- Bỏ qua các box cao dưới 16 px khiến số đo không phản ánh xe rất xa, đúng loại model yếu nhất.
- Nhãn tham chiếu do model tạo và chưa được người rà (xem ca frame_0150 ở mục 2). AP50 đo mức khớp với một
  model khác, không đo mức đúng với con người. Nhãn tôi gán theo guideline (ví dụ thêm xe bị lóa, không gán vệt
  phản chiếu) có thể khác phong cách tham chiếu, và làm AP50 thấp đi dù nhãn đúng hơn.

**Nếu AP50 giảm, tôi kiểm tra trước khi train thêm**:

1. Phân bố conf của model mới ở ngưỡng thấp (0.01), để phân biệt conf sụp với định vị sai.
2. Đường loss và mAP val, cùng số bước cập nhật thực tế (số ảnh / batch × epoch).
3. File nhãn nạp vào có đúng class 0 và tọa độ chuẩn hóa không, và số box train có khớp `round1_diff.md` không.
4. So sánh trực quan vài ảnh test giữa tham chiếu và nhãn của mình, xem phong cách box (ôm sát hay gồm vệt
   đèn) có lệch nhau không.
