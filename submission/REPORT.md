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

- Thời gian (từ `time_log.csv`, **cả hai đều là ước lượng, không bấm đồng hồ**): gán tay 15 phút / 4 ảnh ≈ **3.8 phút/ảnh** (15/44 ≈ 0.34 phút/vật) vs duyệt có pre-label 87 phút / 24 ảnh ≈ **3.6 phút/ảnh** (87/603 ≈ 0.14 phút/vật). Thiên lệch: tôi gán gold trước nên đã quen cảnh; ảnh batch dày hơn gold (≈ 25 vật/ảnh so với ≈ 11 vật/ảnh); và lúc duyệt Claude đã đánh dấu lỗi trước, nên thời gian duyệt thấp hơn một người tự duyệt. Vì vậy so sánh tốc độ chỉ là tham khảo.
- Kết luận: **có, nhưng chỉ cho các lớp COCO phổ biến** (`person`, `motorbike`, `car`): recall trên gold 37/41 = 90.2% và thời gian trên mỗi vật thấp hơn gán tay (0.14 so với 0.34 phút/vật), với điều kiện phải xoá kỹ box thừa (precision trên gold chỉ 52.9%). **Không đáng tin cho `three_wheeler`**: 19/43 box YOLOE bị xoá, nên chỉ dùng làm gợi ý và phải duyệt từng box.
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. Ẩn nhãn máy và đếm vật trước ở **24/24 ảnh**, ghi số đếm mỗi ảnh vào sổ nháp; sau duyệt so số đếm với số box trong `batch_reviewed.json` (kiểm bằng sổ ghi: đủ 24 dòng).
  2. Với lớp ngưỡng thấp (`person` = 0.15): lọc các box có `conf` < 0.3 và kiểm **từng box**; kiểm bằng `review_stats`: tỉ lệ deleted của `person` ở batch phải gần tỉ lệ box sai của máy trên gold, không phải 7/375 như lần này.
  3. Vật nhỏ (< 40 px) hoặc sát mép ảnh: **phóng to ≥ 200%** trước khi chấp nhận box; kiểm bằng số bẫy loose / missing lọt ở vật nhỏ, mép ảnh (lần này lọt đúng 2 bẫy loại đó, mục tiêu 0).

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| <<img_0xx>> | <<>> | <<>> | <<>> | <<>> | <<>> |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu? <<>>
- *(Tuỳ chọn)* Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline: <<>>

## 6. Calibration (Part 6)

- Số quyết định: <<>> (`calibration_decisions.csv`); quan trọng nhất (dẫn `CAL-xx`): <<>>
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: <<>>
- Nguyên nhân gốc: <<guideline mơ hồ / chưa nắm quy tắc / máy>>

## 7. QC và quyết định (Part 6)

- Mẫu: <<n>> ảnh (<<m>> ảnh `can_review` + <<k>> ngẫu nhiên, seed 42); người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt).
- Lỗi theo **đối tượng**: <<.../... = ...%>> · theo **ảnh**: <<.../... = ...%>> (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: <<>>
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): <<>>
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): <<>>

**Quyết định: <<Release / Hold / Rework>>**, ngày <<YYYY-MM-DD>>. Lý do (đi theo Hold → Rework → Release): <<>>
Nếu Rework / Hold, việc cần làm tiếp: <<>>

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài**: dùng Claude (claude.ai)
- **Part 4 có AI hỗ trợ** (thầy cho phép làm riêng): Claude đọc `traps_key.json` **trước khi tôi duyệt** và khoanh 12 bẫy trên ảnh; Claude soát 24 ảnh batch, đánh dấu box trùng / thừa / sai lớp / bỏ sót, tôi sửa trong CVAT theo đó và tự quyết các ca mơ hồ. Vì vậy tỉ lệ bắt bẫy và thời gian `batch_review` không phản ánh một lần tự duyệt. Claude cũng soạn nháp mục 2–4 từ `review_stats`; tôi đối chiếu từng số với `review_stats.md`.
- Gold có bị nhiễm không: có, một phần. Lúc gán tay 4 ảnh gold tôi không xem pre-label. Sau khi chạy `threshold_sweep` lần đầu (person chỉ đạt recall tối đa 71%), nhờ phân tích so khớp tôi biết được các box gold mà máy "sót" và đã sửa: (1) img_012 chiếc SUV bị cắt mép trên gán nhầm `person` → đổi thành `car`; (2) img_036 `ignore_region` phủ lên 3 người đã vẽ riêng → thu nhỏ vùng ignore, giữ các người; (3) tick `can_review` cho người lái tuk-tuk img_012. Trong lúc sửa tôi lỡ đổi xe bán tải img_001 từ `car` sang `truck` theo nhãn máy, sau đó đổi lại `car` theo guideline mục 2 (ví dụ thiên kiến tự động hóa của chính tôi). Các sửa đổi dựa trên ảnh gốc và guideline v1.0, nhưng việc biết chỗ cần xem đến từ kết quả máy, nên số đo máy trên gold có thể lạc quan hơn.
- Máy / GPU / CVAT dùng: Windows, NVIDIA RTX 4050 Laptop 6 GB (CUDA), Python 3.13, torch 2.6.0+cu124, ultralytics 8.4.215, CVAT v2.81.0 chạy Docker. Vướng mắc: PowerShell chặn `Activate.ps1` (sửa bằng `Set-ExecutionPolicy -Scope Process Bypass`); tài khoản CVAT cũ đã tồn tại nên dùng lại.
- Dùng cache `data/cache/` thay vì tự chạy? Không, tự chạy toàn bộ trên GPU máy mình.
- Lỗi / bug script đã gặp (nếu có): không gặp bug script.
- Thời gian: quên bấm đồng hồ ở cả Part 1 và Part 4. `gold_manual` ghi theo mốc nhớ lại (18:50–19:05); `batch_review` ước lượng từ mốc mở Job (10:21) đến lúc xong (11:58), trừ ~10 phút chờ AI. Cả hai không phải số đo bằng đồng hồ.
- Một điều bạn làm khác lần sau: <<>>
