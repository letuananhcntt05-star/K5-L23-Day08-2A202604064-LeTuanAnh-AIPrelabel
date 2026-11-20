# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202604064 - Lê Tuấn Anh

> Chỉ ghi phần thay đổi so với v1.0. Mỗi ca khó có ảnh ví dụ và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 4 | Người ngồi trong xe ba bánh có mui, chỉ thấy thấp thoáng | Gán `person` khi thấy rõ thân hoặc chân (khoảng ≥ 20% người); box chỉ bao phần nhìn thấy, không đoán phần bị mui che. Ca khó vẫn tick `can_review` | img_012 (người lái tuk-tuk nhìn qua khung cửa) | CAL-03 |
| 6 | Box máy chồng nhau quá nhiều | **Một vật chỉ có một box.** Giữ box ôm sát cả vật (với xe ba bánh: từ đỉnh mui tới đáy bánh), xoá các box còn lại | img_003 (3 box trên xe ba gác), img_010 (2 box trên xích lô), img_013 (2 box trên tuk-tuk) | CAL-05 |
| 7 | Mức `conf` tối thiểu để chấp nhận box máy không cần sửa | **Không đặt.** Mọi box máy đều phải kiểm; box có `conf` < 0.3 phải kiểm từng box, vì `conf` không phải xác suất đúng | img_005, img_024 (nhiều box `person` điểm thấp) | CAL-06 |

**Chưa chốt** (chưa gặp đủ ca rõ trong 28 ảnh đã gán để có ảnh ví dụ): câu 1 (xe đạp điện), câu 2 (khối xe máy đỗ dày gần camera, ví dụ img_026 cần thêm thảo luận), câu 3 (hàng nhô ra), câu 5 (kính xe trong mask). Giữ nguyên v1.0 và tick `can_review` khi gặp.

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| img_013, img_004 | Tuk-tuk | YOLO26 gọi `car` (0.41; 0.37 / 0.34) | `three_wheeler` | Mục 5: xe ba bánh có mui (tuk-tuk) **luôn** là `three_wheeler`, kể cả khi máy gợi ý `car` / `truck` điểm cao (CAL-01) |
| img_001 | Xe bán tải 2 cabin, thùng hở | YOLO26 gọi `truck` 0.88 | `car` | Mục 5: bán tải (pickup) là `car`; `truck` chỉ cho xe có buồng lái riêng + thùng hàng lớn (CAL-02) |
| img_014, img_009 | Quang gánh, kệ hàng, ô che sạp | YOLOE gọi `three_wheeler` | Xoá box | Mục 5: chỉ gán `three_wheeler` khi **thấy bánh xe và khung xe** (CAL-04) |
| img_031, img_018 | Người ngồi sát mép phải; xe máy rất nhỏ (~40 px) | Bẫy `missing` / `loose` bị lọt khi duyệt | Không đổi quy tắc nhãn | Đây là **lỗi người duyệt**, không phải guideline mơ hồ: quy tắc 4.1 đã rõ. Thêm vào checklist: phóng to ≥ 200% với vật < 40 px hoặc sát mép (CAL-07) |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt câu hỏi mở 4, 6, 7; thêm 3 quy tắc lớp (tuk-tuk luôn `three_wheeler`, bán tải là `car`, `three_wheeler` phải thấy bánh + khung); ghi nhận bẫy lọt là lỗi người duyệt, không đổi quy tắc | CAL-01 … CAL-07 |
