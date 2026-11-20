# Báo cáo Day 08 — 2A202604064 - Lê Tuấn Anh

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person | YOLO26m | 0.15 | `threshold_sweep`: ngưỡng cao nhất (≤ 0.5) còn đạt recall ≥ 90%: recall 16/17 = 94%, precision 16/35 = 46%. Ở 0.3 recall chỉ còn 15/17 = 88%. r02/r03: ngưỡng 0.5 làm mất người thật (img_011 người cắt mép phải chỉ 0.49) |
| motorbike | YOLO26m | 0.3 (mặc định) | Không ngưỡng nào ≤ 0.5 đạt recall 90% (tối đa 8/9 = 89% ở ngưỡng 0.10); ở 0.3: recall 7/9 = 78%, precision 7/11 = 64% |
| car | YOLO26m | 0.3 (mặc định) | Không đạt 90% ở mọi ngưỡng (tối đa 7/8 = 88%): vật sót là xe bán tải img_001 bị YOLO26 gọi `truck 0.88` (sai lớp, không phải do ngưỡng) |
| bicycle / bus / truck | YOLO26m | 0.3 (mặc định) | 0 vật trong 4 ảnh gold: không có dữ liệu để chọn ngưỡng riêng |
| three_wheeler | YOLOE-26s, prompt mặc định từ taxonomy: `tuk-tuk`, `cyclo`, `cargo tricycle` | 0.15 | Recall 7/7 = 100%, precision 7/9 = 78%; ở 0.3 recall rơi còn 5/7 = 71%. YOLO26 không có lớp này (r02, r05: tuk-tuk bị gọi `car`); r04: YOLOE bắt đúng tuk-tuk img_002 (0.93) |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "person=0.15,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json` → 725 box cho 28 ảnh (person 430, motorbike 154, car 65, three_wheeler 52, bicycle 13, truck 7, bus 4).
- **Phiên bản SAM đã chọn: SAM 2.1 base (`sam2.1_b.pt`)**. Lý do: có sẵn trong bộ `--set core`, chạy được trên GPU RTX 4050 6 GB (r05: 1.86 s/ảnh gồm cả YOLO26 + SAM cho 16 mask), không cần xin quyền như SAM 3; khi box đúng thì mask bám sát biên (r05: img_004 người áo tím, tuk-tuk xanh lá).
- Lỗi **không sửa được bằng đổi ngưỡng**: sai lớp. Tuk-tuk bị YOLO26 gọi `car` (r05: img_013 `car 0.41`, img_004 `car 0.37`/`0.34`); xe xanh mép trái img_011 là `truck 0.77` ở cả ngưỡng 0.25 lẫn 0.5 (r02, r03); xe bán tải img_001 là `truck 0.88`. Cách sửa: dùng YOLOE cho `three_wheeler` và gộp YOLOE **trước** YOLO26 với `--iou 0.6`; lần gộp đã loại 11 box `car`, 6 `truck`, 2 `bus`, 1 `bicycle` của YOLO26 chồng lên box `three_wheeler`. Sai lớp còn lại (bán tải → `truck`) phải sửa tay khi duyệt.
- Box **điểm cao mà sai**: img_001 `truck 0.88` (là bán tải, theo guideline mục 2 phải là `car`); img_011 `truck 0.77` (xe xanh mép trái, nghi songthaew bán tải). Box **điểm thấp mà đúng**: img_011 `person 0.49` (người bị cắt mép phải, r02), img_011 `motorbike 0.52`.
- Giới hạn: ngưỡng được chọn và chấm trên **cùng 4 ảnh gold** (person 17, motorbike 9, car 8, three_wheeler 7 vật; bicycle / bus / truck 0 vật) nên precision / recall của máy trên gold là **lạc quan**; mẫu nhỏ nên ngưỡng chỉ là ước lượng thô, và chưa kiểm được ngưỡng cho 3 lớp không có vật trong gold.

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 591 | 591/628 = 94.1% | 628 box máy |
| edited | 4 | 4/628 = 0.6% | 628 box máy |
| deleted | 33 | 33/628 = 5.3% | 628 box máy |
| added | 8 | 8/603 = 1.3% | 603 đối tượng sau duyệt |

Đã loại 22 đối tượng quanh 12 bẫy khỏi bảng này, vì đó là lỗi nhân tạo, không phải lỗi thật của máy. Trong deleted + added có 4 trường hợp là **đổi lớp** (box đúng chỗ, sai nhãn), nên số vật máy **sót thật** chỉ khoảng 8 − 4 = 4.

- Lớp nào máy tệ nhất? **`three_wheeler` (YOLOE)**: 19/43 box máy bị xoá (44%), 3/43 phải sửa. Nguyên nhân: nhiều box chồng lên cùng một xe (img_003, img_008, img_010, img_013, img_017, img_023, img_035), box trùm 2–3 xe một lúc (img_008, img_017, img_023), và gọi nhầm vật khác là xe ba bánh (quang gánh và kệ hàng ở img_014, ô che sạp ở img_009, xe máy chở rau ở img_009). Lớp `bus` có 3 added: xe buýt ở img_013, img_018 bị máy gán sai lớp nên phải vẽ lại / đổi sang `bus`.
- So với gold: precision / recall của **máy** = 37/70 = **52.9%** (KTC 95%: 41–64%) / 37/41 = **90.2%** (KTC 95%: 77–96%) (mẫu số: 41 đối tượng gold, 70 box máy trên 4 ảnh gold). Hai con số này **lạc quan** vì ngưỡng được chọn trên chính 4 ảnh gold.

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: **10/12 = 83.3%** (KTC 95%: 55–95%); theo loại: loose 2/3, extra 3/3, missing 2/3, wrong_class 3/3. Lọt: **img_031 (missing)** – người ngồi sát mép phải chưa được vẽ khớp; **img_018 (loose)** – box xe máy rất nhỏ (~40 px) chưa kéo đủ sát. **Lưu ý trung thực:** tôi đã xem `traps_key.json` trước khi duyệt (theo hướng dẫn riêng của thầy) và Claude đã khoanh 12 bẫy trên ảnh, nên con số 83.3% **không đo** khả năng tự phát hiện lỗi của tôi. Dù biết trước đáp án vẫn lọt 2 bẫy, cả hai đều là vật **nhỏ hoặc sát mép ảnh**: tôi đã không phóng to khi làm bước 4 (lỏng) và bước 1 (bỏ sót) với những vật này.
- Accepted cao mà bắt bẫy (có đáp án) cũng chưa đủ nói gì về cách duyệt: trên gold, máy có 33/70 box sai (precision 52.9%), nhưng ở batch tôi chỉ xoá 33/628 = 5.3% box máy, riêng `person` (ngưỡng 0.15) chỉ xoá 7/375. Nếu batch có tỉ lệ box sai gần với gold, thì còn **nhiều box thừa đã bị giữ lại**, nhất là ở 6 ảnh dày xe (005, 018, 021, 024, 028, 029): đây là dấu hiệu chấp nhận nhãn máy quá dễ (thiên kiến tự động hóa). Đây là suy luận, vì gold chỉ có 4 ảnh và cảnh khác batch.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`, **cả hai đều là ước lượng, không bấm đồng hồ**): gán tay 15 phút / 4 ảnh ≈ **3.8 phút/ảnh** (15/41 ≈ 0.37 phút/vật) vs duyệt có pre-label 87 phút / 24 ảnh ≈ **3.6 phút/ảnh** (87/603 ≈ 0.14 phút/vật). Thiên lệch: tôi gán gold trước nên đã quen cảnh; ảnh batch dày hơn gold (603/24 ≈ 25 vật/ảnh so với 41/4 ≈ 10 vật/ảnh); và lúc duyệt Claude đã đánh dấu lỗi trước, nên thời gian duyệt thấp hơn một người tự duyệt. Vì vậy so sánh tốc độ chỉ là tham khảo.
- Kết luận: **có, nhưng chỉ cho các lớp COCO phổ biến** (`person`, `motorbike`, `car`): recall trên gold 37/41 = 90.2% và thời gian trên mỗi vật thấp hơn gán tay (0.14 so với 0.37 phút/vật), với điều kiện phải xoá kỹ box thừa (precision trên gold chỉ 52.9%). **Không đáng tin cho `three_wheeler`**: 19/43 box YOLOE bị xoá, nên chỉ dùng làm gợi ý và phải duyệt từng box.
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. Ẩn nhãn máy và đếm vật trước ở **24/24 ảnh**, ghi số đếm mỗi ảnh vào sổ nháp; sau duyệt so số đếm với số box trong `batch_reviewed.json` (kiểm bằng sổ ghi: đủ 24 dòng).
  2. Với lớp ngưỡng thấp (`person` = 0.15): lọc các box có `conf` < 0.3 và kiểm **từng box**; kiểm bằng `review_stats`: tỉ lệ deleted của `person` ở batch phải gần tỉ lệ box sai của máy trên gold, không phải 7/375 như lần này.
  3. Vật nhỏ (< 40 px) hoặc sát mép ảnh: **phóng to ≥ 200%** trước khi chấp nhận box; kiểm bằng số bẫy loose / missing lọt ở vật nhỏ, mép ảnh (lần này lọt đúng 2 bẫy loại đó, mục tiêu 0).

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask có gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_013 | SAM 2.1 base (`sam2.1_b.pt`) | 3 (2 dương: thân xe, mui; 1 âm: người lái) | **Có, một phần**: 1 click âm chỉ loại được phần áo người lái; chân, bàn chân, tay và mặt người lái vẫn bị tô | Mask bám sát mui, bánh, cản sau; nhưng **lan sang người và ghế ở quán phía sau** khi nhìn qua khung xe hở | `tuktuk_img013_sam2.jpg` |
| img_004 | SAM 2.1 base (`sam2.1_b.pt`) | 3 (2 dương: thân xe, sàn; 1 âm: người đứng giữa 2 xe) | **Không**: xe không có người lái; click âm loại được người đứng cạnh. Bình gas / thùng đỏ gắn trên xe được tô, đúng vì là một phần của xe | Mask trọn mui, thân, 3 bánh; lấn một ít xuống bóng đổ dưới gầm trước (khoảng x 650–740, y 545–565) | `tuktuk_img004_sam2.jpg` |
| img_017 | SAM 2.1 base (`sam2.1_b.pt`) | 3 (2 dương: ghế, thân xe; 1 âm) | **Không**: xe không có người. Click âm (645, 355) lại rơi đúng lên mui xe nên không loại trừ được gì: toạ độ mẫu không khớp vật cần loại | Mask bám tốt mui, thân, bánh; tuk-tuk phía sau không bị tô; vùng quầy hàng nhìn xuyên qua khung hở bị tô một phần | `tuktuk_img017_sam2.jpg` |

- Đối chiếu guideline mục 4.2: quy tắc "polygon chỉ bao phần nhìn thấy của vật" và "người là `person` riêng" giúp xác định mask img_013 **sai**: mask xe không được gồm người lái và những gì nhìn xuyên qua khung xe. Quy tắc còn thiếu: với xe có khung hở (tuk-tuk, xích lô), vùng nhìn xuyên qua khung có thuộc mask xe không. Đây là câu hỏi mở số 5 (kính / cửa sổ xe) mà v1.1 chưa chốt. Bài học: với xe có người lái, một click âm là không đủ; cần đặt click âm lên đầu, thân và chân người lái, và thêm click âm vào vùng nền nhìn qua khung.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: không dùng được, vì CVAT Docker cài mặc định chưa có serverless (Nuclio), menu Models trống.

## 6. Calibration (Part 6)

- Số quyết định: **7** (`calibration_decisions.csv`): 3 `loi_may`, 3 `cau_hoi_mo`, 1 `bay_lot`. Quan trọng nhất: **CAL-05** (một vật chỉ có một box), vì box trùng là lỗi máy phổ biến nhất ở lớp `three_wheeler` (19/43 box YOLOE bị xoá) và vẫn còn sót trong nhãn đã nộp (img_010, img_013).
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: chốt câu hỏi mở 4 (người trong tuk-tuk), 6 (box chồng nhau), 7 (không đặt mức `conf` tối thiểu); tuk-tuk luôn là `three_wheeler`; bán tải là `car`; `three_wheeler` phải thấy bánh và khung xe. Câu 1, 2, 3, 5 chưa chốt vì chưa có ảnh ví dụ rõ.
- Nguyên nhân gốc: **máy** (CAL-01, 02, 04: sai lớp của YOLO26, nhận nhầm của YOLOE); **guideline mơ hồ** (CAL-03, 05, 06: câu hỏi mở của v1.0); **tôi làm sai** (CAL-07: bẫy lọt do không phóng to vật nhỏ, vật sát mép; và ở gold img_001 tôi từng đổi bán tải theo máy). Với lỗi của tôi thì không đổi quy tắc nhãn, chỉ siết checklist.

## 7. QC và quyết định (Part 6)

- Mẫu: **5** ảnh (0 ảnh `can_review` vì không box nào được tick + 5 ngẫu nhiên, `qc_sample.py --n 5 --seed 42`), 117 đối tượng: img_003, img_006, img_014, img_031, img_035.
- Lỗi theo **đối tượng**: **6/117 = 5.1%** · theo **ảnh**: **3/5 = 60%** (img_003, img_031, img_035).
- Top lỗi: **sai lớp** 2 (img_003: 2 xe tải ở xa bị gán `three_wheeler`), **sai vị trí / kích thước** 2 (img_003 box xe ba gác thiếu phần mui; img_035 box xích lô thiếu phần mui), **thừa** 1 (img_031 áo khoác treo bị gán `person`), **bỏ sót** 1 (img_031 người ngồi sát mép phải, cũng là bẫy lọt).
- **Giới hạn mẫu nhỏ**: với 3/5 ảnh lỗi, cận trên 95% (Wilson) của tỉ lệ ảnh lỗi thật là 88% (`qc_report_auto.md`). Kể cả nếu mẫu 5 ảnh không có lỗi nào, cận trên vẫn khoảng 43%, nên một mẫu nhỏ như vậy **không bao giờ chứng minh được** "dưới 2%". Ở đây thì không cần chứng minh, vì mẫu đã vượt ngưỡng rõ.
- Script kiểm được / không kiểm được: script chỉ đếm theo phiếu tôi điền bằng mắt. Nó không tự biết box có ôm sát không, không thấy vật mà cả máy lẫn tôi cùng bỏ sót, và không kiểm được tôi có làm đúng checklist hay không. Ngoài ra lúc duyệt có AI chỉ lỗi trước, nên mẫu QC đo chất lượng nhãn, không đo khả năng tự duyệt của tôi.

**Quyết định: Rework**, ngày 2026-11-20. Lý do (đi theo Hold → Rework → Release): **không Hold**, vì các câu hỏi mở ảnh hưởng tới lô này (4, 6, 7) đã chốt trong v1.1 và dữ liệu không hỏng; **Rework**, vì ảnh có lỗi 3/5 = 60% > 2%, và lỗi `three_wheeler` lặp lại (sai lớp + thiếu mui chiếm 4/6 lỗi). Chưa đủ điều kiện Release.
Việc cần làm tiếp: (1) sửa toàn bộ box `three_wheeler` theo CAL-05 (một vật một box, trùm từ mui tới bánh), ưu tiên img_003, img_008, img_010, img_013, img_017, img_023, img_035; (2) đổi 2 xe tải ở xa img_003 sang `truck`; (3) xoá box thừa và vẽ thêm người img_031; (4) duyệt lại 6 ảnh dày xe với `ignore_region` và kiểm từng box `person` có `conf` < 0.3 (CAL-06); (5) export lại, chạy lại `review_stats` và lấy mẫu QC mới.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài**: dùng Claude (claude.ai) để đọc hiểu đề và hướng dẫn thứ tự lệnh
