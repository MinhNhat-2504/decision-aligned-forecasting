# Dự báo nhu cầu gắn với quyết định cho giao hàng chặng cuối

Mã nguồn của bài báo *Forecast Value Is a Property of the Decision: Calibrated Quantile
Demand Forecasting for Urban Last-Mile Capacity and Routing*.

> **Bài báo đang trong quá trình phản biện.** Repo này chỉ chứa mã nguồn và kết quả số;
> bản thảo chưa được đưa lên. Khi bài được công bố, chúng tôi sẽ bổ sung liên kết DOI và
> thông tin trích dẫn vào đây.

## Bài này nói gì

Phần lớn nghiên cứu dự báo mặc định rằng sai số thấp hơn thì giá trị vận hành cao hơn.
Với bài toán cam kết năng lực phục vụ, suy luận đó không an toàn.

Mức cam kết tối ưu chi phí là một **phân vị** của nhu cầu, và bậc của phân vị đó chính là
**mức dịch vụ mục tiêu**. Nghĩa là khi một doanh nghiệp lập kế hoạch năng lực từ dự báo
trung vị, họ đã ngầm chọn mức dịch vụ 50% — thường là không hề chủ ý.

Chúng tôi đưa **cùng một bộ dự báo** vào hai quyết định tiêu thụ hai đại lượng khác nhau,
và thu được hai kết luận trái ngược:

| Quyết định | Cần gì | Kết quả |
|---|---|---|
| Cam kết năng lực | phân vị đuôi đã hiệu chuẩn | giảm **42–51%** chi phí, tỷ lệ đáp ứng tăng từ 94,9% lên 98,9% |
| Chọn vùng phục vụ | chỉ cần một thứ tự xếp hạng | mô hình học máy hơn trung bình lịch sử **dưới 0,15 điểm phần trăm** |

Một số kết quả **âm** cũng được báo cáo đầy đủ, vì chúng quan trọng không kém: mô hình có
sai số thấp nhất không phải mô hình rẻ nhất; không kiến trúc nào xếp hạng ổn định giữa hai
chế độ nhu cầu; và không quan hệ đồ thị thiết kế thủ công nào cho lợi ích tái lập được.

Thứ chuyển giao được là **tầng quyết định** — đầu ra phân vị đơn điệu cộng hiệu chuẩn
conformal — chứ không phải kiến trúc mạng.

## Cấu trúc

```
src/          43 script Python: dựng dữ liệu, mô hình, thí nghiệm, phân tích
outputs/      bảng kết quả (.csv/.json) cho bộ taxi vàng
outputs_green/ tương tự cho bộ taxi xanh
figures/      hình trong bài, đều sinh từ code
run_*.sh      các kịch bản chạy theo giai đoạn
```

Vài file đáng đọc trước:

- `src/config.py` — toàn bộ tham số: mốc chia thời gian, tầm dự báo, cấu hình đồ thị
- `src/dataset.py` — cửa sổ trượt, chia theo thời gian, học trên phần dư mùa vụ
- `src/models.py` — các mạng nền và đầu phân vị đơn điệu
- `src/train_eval.py` — vòng huấn luyện và hiệu chuẩn conformal
- `src/operations.py` — quyết định newsvendor và chọn vùng
- `src/audit_original.py` — xem mục *Vì sao có repo này* bên dưới

## Chạy lại

### 1. Cài môi trường

```bash
pip install -r requirements.txt
```

PyTorch cần cài riêng cho đúng bản CUDA của máy bạn, xem
[hướng dẫn chính thức](https://pytorch.org/get-started/locally/). Thí nghiệm trong bài
chạy trên `torch 2.5.1+cu121`.

### 2. Lấy dữ liệu

Repo **không kèm dữ liệu**. Tất cả đều là dữ liệu công khai của bên thứ ba, mỗi bộ có giấy
phép riêng nên không được phát tán lại. Các script tải nằm sẵn trong `src/`:

| Bộ dữ liệu | Script | Dùng để |
|---|---|---|
| NYC TLC (taxi vàng, xanh) | `data_build.py`, `data_build_green.py` | dữ liệu nhu cầu chính |
| Porto taxi | `external_porto.py` | kiểm chứng ở thành phố khác |
| Olist (TMĐT Brazil) | `fetch_olist_orders.py` | dòng đơn hàng bưu kiện thật |
| Amazon Last Mile | `fetch_amazon_parcel.py` | sức chứa xe và tuyến giao thật |
| Meituan | `fetch_meituan_delivery.py` | chuỗi cùng-một-đơn từ tạo đến giao |
| Cainiao-LaDe | `fetch_lade_parcel_gps.py` | 184 ngày lịch sử bưu kiện kèm sự kiện GPS |

Dữ liệu NYC gốc lấy từ một kho nén; trỏ đường dẫn qua biến môi trường:

```bash
export GAT_ZIP="/duong/dan/toi/GAT_LastMile_Delivery 3.zip"
```

### 3. Chạy pipeline

```bash
bash run_all.sh          # benchmark, ablation, quyết định vận hành, hình vẽ
bash run_gpu.sh          # thêm DCRNN/Graph WaveNet, và toàn bộ bộ taxi xanh
bash run_3a.sh           # hiệu chuẩn Mondrian (conformal theo nhóm)
```

Đổi bộ dữ liệu bằng biến môi trường:

```bash
DATASET=green python -u src/train_eval.py --seeds 3
```

Toàn bộ chuỗi thí nghiệm mất khoảng **12 giờ trên một GPU**. Các bước phân tích hạ nguồn
(`dm_test.py`, `operations.py`, `ksweep.py`, `multiplicity.py`) chạy trên CPU trong vài phút.

## Vài điều nên biết trước khi đọc số

**Không tái lập bit-chính-xác.** Các nhân rút gọn trên GPU không tất định, nên chạy lại
cùng một cấu hình với cùng seed vẫn ra kết quả hơi khác. Chúng tôi **đo** biên độ đó
(`src/nondeterminism.py`): khoảng **0,83%** cho MAE trên bộ vàng và **0,70%** trên bộ xanh.
Con số này là **ngưỡng phân giải** của cả nghiên cứu — chênh lệch nhỏ hơn nó thì không thể
quy cho can thiệp nào. Chúng tôi dùng chính ngưỡng này để chấm lại các tuyên bố của mình,
và nó khiến một kết quả của chúng tôi phải hạ xuống mức "chỉ tạm coi là có".

**Không có rò rỉ nhãn.** Chia dữ liệu theo thời gian, không xáo trộn, mẫu được gán vào tập
theo thời điểm của nhãn. Tập hiệu chuẩn conformal nằm ở **giai đoạn sau** tập dừng sớm và
tách rời khỏi nó.

**Kết quả âm được giữ nguyên.** Repo này tái lập cả những kết quả không ủng hộ giả thuyết
ban đầu. Đó là chủ ý.

## Vì sao có repo này

Dự án khởi đầu từ một pipeline có sẵn. Bước đầu tiên là kiểm toán nó
(`src/audit_original.py`), và kết quả kiểm toán cho thấy:

```
corr(nhu cầu, historical_trip_count) = 1.0000000000
R² của hồi quy tuyến tính            = 1.0
```

Đặc trưng đầu vào **chính là nhãn**. Mô hình đang dự đoán thứ nó đã được cho biết. Toàn bộ
số liệu của phiên bản cũ vì thế vô nghĩa, và pipeline được xây lại từ đầu.

Script kiểm toán vẫn nằm trong repo, chạy được, để bất kỳ ai cũng tự xác minh lại điều đó.

## Giấy phép

Mã nguồn phát hành theo giấy phép MIT.

Dữ liệu **không** thuộc phạm vi giấy phép này. Mỗi bộ dữ liệu giữ giấy phép gốc của nó
(NYC TLC, Amazon, Meituan, Olist, Cainiao-LaDe, Porto). Vui lòng đọc điều khoản của từng
bên trước khi sử dụng.

## Trích dẫn

Bài đang phản biện. Sẽ cập nhật khi có DOI.
