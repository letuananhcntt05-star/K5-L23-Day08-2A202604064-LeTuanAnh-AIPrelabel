# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 628; đối tượng sau duyệt: 603; ngưỡng accepted: IoU >= 0.9
- Đã **loại 22 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 591 | 591/628 = 94.1% | 628 box máy |
| edited (sửa) | 4 | 4/628 = 0.6% | 628 box máy |
| deleted (xoá) | 33 | 33/628 = 5.3% | 628 box máy |
| added (vẽ thêm) | 8 | 8/603 = 1.3% | 603 đối tượng sau duyệt |

Trong số deleted + added, **4 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 4.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 367 | 1 | 7 | 1 |
| bicycle | 6 | 0 | 2 | 1 |
| motorbike | 136 | 0 | 3 | 1 |
| car | 54 | 0 | 0 | 0 |
| bus | 3 | 0 | 0 | 3 |
| truck | 4 | 0 | 2 | 0 |
| three_wheeler | 21 | 3 | 19 | 2 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 41 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 37 | 33 | 4 | 52.9% (KTC 95%: 41–64%) | 90.2% (KTC 95%: 77–96%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 10/12 = 83.3% (KTC 95%: 55–95%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - loose: 2/3
  - extra: 3/3
  - missing: 2/3
  - wrong_class: 3/3
  - lọt: img_031.jpg (missing), img_018.jpg (loose)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
