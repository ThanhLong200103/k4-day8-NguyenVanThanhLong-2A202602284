# Quét độc lập trước khi xem pre-label

Frame: frame_0099.jpg trong `to_label/round1/images/train/`

Số xe nhìn thấy bằng mắt: khoảng 21 xe (cảnh cao tốc ban đêm; nhiều xe ở xa chỉ thấy được qua đèn pha/đèn hậu)

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe:
1. Giữa mép dưới ảnh (khoảng x≈510–630, y≈575–720): xe SUV tối màu, lớn, đang chạy về phía camera, bật đèn pha trắng, bị cạnh dưới ảnh cắt mất một phần. Đèn chói và thân xe bị cắt nên dễ bị bỏ sót.
2. Làn giữa, chính giữa ảnh (khoảng x≈735–830, y≈430–490): sedan tối màu đang chạy về phía camera, đèn pha chói lóa làm mờ hình dạng thân xe, nên AI dễ bỏ qua hoặc chỉ khoanh quanh vùng đèn.

Chạy `python3 tools/lock_blind.py` ngay sau khi điền. Sau đó giữ file này nguyên vẹn.
