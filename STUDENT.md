# Thông tin học viên

- Họ và tên: Lê Tuấn Anh
- MSHV: 2A202604064
- Lớp (L2 / L3 / khác): L2B
- Tài khoản GitHub: letuananhcntt05-star
- Repo bài nộp (URL): https://github.com/letuananhcntt05-star/K5-L23-Day08-2A202604064-LeTuanAnh-AIPrelabel
- Commit chốt bài (hash): 7b854a3192411b83f23ea3697c968086b61b88dc
- Máy: Windows, NVIDIA GeForce RTX 4050 Laptop GPU 6 GB (CUDA, torch 2.6.0+cu124)
- CVAT (Docker): v2.81.0

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 591/628 = 94.1% / 4/628 = 0.6% / 33/628 = 5.3% (mẫu số: 628 box máy); added: 8/603 = 1.3% (mẫu số: 603 đối tượng sau duyệt)
- So với gold: precision / recall của **máy** 37/70 = 52.9% / 37/41 = 90.2% (4 ảnh gold, lạc quan vì ngưỡng chọn trên chính gold)
- Bắt bẫy: 10/12 = 83.3% (đã xem `traps_key.json` trước khi duyệt, theo hướng dẫn riêng của thầy, nên không đo khả năng tự phát hiện)
- Thời gian (ước lượng, không bấm đồng hồ): gán tay 15/4 ≈ 3.8 phút/ảnh; duyệt có pre-label 87/24 ≈ 3.6 phút/ảnh
- Lỗi QC: 60% theo ảnh (3/5), 5.1% theo đối tượng (6/117)
- **Quyết định:** Rework — lý do một dòng: ảnh có lỗi 3/5 = 60% > 2% và lỗi `three_wheeler` lặp lại (box thiếu mui, sai lớp)
