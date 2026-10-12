# Dự báo nhu cầu gắn với quyết định cho giao hàng chặng cuối

Mã nguồn và kết quả số của bài báo

> **Forecast Value Is a Property of the Decision: Calibrated Quantile Demand Forecasting
> for Urban Last-Mile Capacity and Routing**
> Kien Pham Xuan, Nhat Trinh Ngoc Minh, Linh Nguyen Thi My
> *International Journal of Analysis and Applications*, đã được chấp nhận năm 2026 (đang chờ xuất bản, mã bài ijaa5688).

DOI sẽ được bổ sung khi bài được xuất bản.
Bản tiếng Anh của file này là [README.md](README.md).

## Bài này nói gì

Phần lớn nghiên cứu dự báo mặc định rằng sai số thấp hơn thì giá trị vận hành cao hơn.
Với bài toán cam kết năng lực phục vụ, suy luận đó không an toàn. Mức cam kết tối ưu chi
phí là một **phân vị** của nhu cầu, và bậc của phân vị đó chính là **mức dịch vụ mục
tiêu**. Một doanh nghiệp lập kế hoạch năng lực từ dự báo trung vị đã ngầm chọn mức dịch
vụ 50%.

Chúng tôi đưa **cùng một bộ dự báo** vào hai quyết định tiêu thụ hai đại lượng khác nhau,
trên 263 vùng của New York và 39,0 triệu chuyến taxi từ hai dịch vụ. Hai quyết định cho
hai kết luận trái ngược.

| Quyết định | Cần gì từ dự báo | Kết quả |
|---|---|---|
| Cam kết năng lực (newsvendor, phạt 9:1) | phân vị đuôi đã hiệu chuẩn | chi phí giảm **42–51%**; tỷ lệ đáp ứng 94,9% → 98,9% (taxi vàng) và 86,8% → 97,4% (taxi xanh) |
| Chọn K vùng phục vụ | chỉ cần một thứ tự xếp hạng | mô hình học máy hơn trung bình lịch sử theo tuần **dưới 0,15 điểm phần trăm** |

Ba phát hiện khác, trong đó có hai kết quả âm, được báo cáo đầy đủ:

- Mô hình có sai số thấp nhất không phải mô hình có chi phí thấp nhất.
- Không kiến trúc nào xếp hạng ổn định giữa hai chế độ nhu cầu, và không quan hệ đồ thị
  thiết kế thủ công nào cho lợi ích tái lập được.
- Thứ chuyển giao được là **tầng quyết định**: đầu ra phân vị đơn điệu cộng hiệu chuẩn
  conformal dùng nguyên vẹn giữa các mạng nền, và tái lập mức giảm chi phí 40–54% trên
  năm bộ dữ liệu ngoài (Porto, Olist, Amazon, Meituan, Cainiao-LaDe).

## Cấu trúc repo

```
src/             43 script Python: dữ liệu, mô hình, thí nghiệm, phân tích
outputs/         bảng kết quả (.csv / .json), taxi vàng và các bộ dữ liệu ngoài
outputs_green/   bảng kết quả, taxi xanh
figures/         hình trong bài, đều sinh từ src/figures.py
figures_green/   các hình tương ứng cho taxi xanh
run_*.sh         các kịch bản chạy theo giai đoạn (bash)
run_*.ps1        các bộ kiểm tra độ vững chạy dài (PowerShell)
```

Các bảng kết quả của bài nằm sẵn trong `outputs/` và `outputs_green/`, nên có thể xem số
mà không cần chạy lại gì.

### Pipeline chính

| File | Vai trò |
|---|---|
| `src/config.py` | toàn bộ tham số: mốc chia thời gian, tầm dự báo, cấu hình đồ thị và huấn luyện |
| `src/data_build.py`, `src/data_build_green.py` | nhu cầu theo giờ theo vùng từ dữ liệu chuyến thô |
| `src/graphs.py` | ba quan hệ đồ thị: địa lý, mạng đường, luồng điểm đi – điểm đến |
| `src/dataset.py` | cửa sổ trượt, chia theo thời gian, học trên phần dư mùa vụ |
| `src/models.py` | GCN, GraphSAGE, GAT, ST-MRGAT, DCRNN, Graph WaveNet, MLP, LSTM, đầu phân vị đơn điệu |
| `src/train_eval.py` | vòng huấn luyện, các mô hình đối chứng, hiệu chuẩn conformal |
| `src/operations.py` | hai quyết định: năng lực kiểu newsvendor và chọn K vùng |

### Các phân tích

| Chủ đề | Script |
|---|---|
| Kiểm định thống kê | `dm_test.py` (Diebold–Mariano), `multiplicity.py` (Holm / Benjamini–Hochberg) |
| Hiệu chuẩn | `mondrian_cqr.py`, `calibration_slices.py` |
| Độ vững của quyết định | `decision_robustness.py`, `decision_selection.py`, `decision_focused.py`, `capacity_matched.py`, `economic_sensitivity.py` |
| Định tuyến | `ksweep.py`, `routing_robustness.py`, `vrp_constrained.py`, `vrp_exact.py`, `execution_deviation.py` |
| Ngưỡng nhiễu | `nondeterminism.py` |
| Diễn giải | `analysis.py` (attention mức quan hệ) |
| Kiểm chứng ngoài | `external_porto.py`, `external_porto_neural.py`, `olist_order_arrival.py`, `amazon_parcel_eval.py`, `amazon_parcel_postprocess.py`, `meituan_end_to_end.py`, `lade_shanghai_capacity.py` |
| Hình và gộp kết quả | `figures.py`, `merge_results.py` |

`external_chicago.py` là một kiểm tra phụ, không được báo cáo trong bài.

## Chạy lại

### 1. Cài môi trường

```bash
pip install -r requirements.txt
```

PyTorch cần cài riêng cho đúng bản CUDA của máy, xem
[hướng dẫn chính thức](https://pytorch.org/get-started/locally/). Thí nghiệm trong bài
chạy trên `torch 2.5.1+cu121`.

### 2. Dữ liệu

Repo không kèm dữ liệu. Tất cả là dữ liệu công khai của bên thứ ba, mỗi bộ có giấy phép
riêng nên không được phát tán lại ở đây.

**Tải bằng script có sẵn trong repo:**

| Bộ dữ liệu | Script | Dùng để |
|---|---|---|
| Porto taxi (UCI) | `external_porto.py` | kiểm chứng ở thành phố khác |
| Olist (thương mại điện tử Brazil) | `fetch_olist_orders.py` | dòng đơn hàng bưu kiện thật |
| Amazon Last Mile Routing Challenge | `fetch_amazon_parcel.py` | tuyến giao và sức chứa xe thật |
| Meituan (INFORMS TSL challenge) | `fetch_meituan_delivery.py` | chuỗi cùng một đơn từ tạo đến giao |
| Cainiao-LaDe | `fetch_lade_parcel_gps.py` | 184 ngày lịch sử bưu kiện kèm sự kiện GPS |

**Phải đặt bằng tay.** Ba nguồn dưới đây chưa có script tải.

| Bộ dữ liệu | Vị trí mà code đọc |
|---|---|
| Taxi vàng NYC, tháng 1–3/2016, bản CSV cũ có toạ độ điểm đón và điểm trả; đa giác các vùng taxi NYC; mạng đường OpenStreetMap của New York | một file zip, đường dẫn đặt qua biến môi trường `GAT_ZIP`; đường dẫn từng file bên trong ghi ở `src/config.py` |
| Taxi xanh NYC, tháng 1–3/2016, file parquet từ [trang dữ liệu chuyến của TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | `data/raw2/green_2016-01.parquet`, `green_2016-02.parquet`, `green_2016-03.parquet` |
| Tuyến giao hàng kế hoạch so với thực tế (bộ dữ liệu lệch tuyến được trích dẫn trong bài) | `data/raw3/Last-mile delivery route deviations dataset planned vs. actual routes/routes_performance.xlsx` |

```bash
export GAT_ZIP="/duong/dan/toi/archive.zip"
```

Bước dựng dữ liệu taxi vàng gán mỗi chuyến vào vùng bằng phép kiểm tra điểm trong đa
giác, nên cần bản dữ liệu cũ có toạ độ.

### 3. Dựng bộ dữ liệu

```bash
python src/data_build.py          # nhu cầu taxi vàng + zones.csv
python src/graphs.py              # ba quan hệ đồ thị
python src/data_build_green.py    # nhu cầu và đồ thị taxi xanh (chạy sau hai bước trên)
```

### 4. Chạy thí nghiệm

```bash
bash run_all.sh      # taxi vàng: benchmark (5 seed), ablation, quyết định, hình, attention
bash run_gpu.sh      # thêm DCRNN và Graph WaveNet; toàn bộ taxi xanh; phân tích hạ nguồn
bash run_gwnq.sh     # Graph WaveNet với đầu phân vị, rồi gộp và phân tích lại
bash run_3a.sh       # hiệu chuẩn conformal theo nhóm (Mondrian)
bash run_taskB.sh    # chọn mô hình theo chi phí quyết định
```

Các bộ kiểm tra độ vững là script PowerShell: `run_robustness.ps1` và
`run_final_jobs.ps1`.

Đổi bộ dữ liệu hoặc chỉ chạy một phần:

```bash
DATASET=green python -u src/train_eval.py --seeds 3
python -u src/train_eval.py --models GraphWaveNet-Q
```

Toàn bộ chuỗi thí nghiệm mất khoảng **12 giờ trên một GPU**. Các bước phân tích hạ nguồn
(`dm_test.py`, `operations.py`, `ksweep.py`, `multiplicity.py`) chạy trên CPU trong vài phút.

## Vài điều nên biết trước khi đọc số

**Không tái lập chính xác từng bit.** Các nhân rút gọn trên GPU không tất định, nên chạy
lại cùng cấu hình với cùng seed vẫn ra kết quả hơi khác. Chúng tôi đo biên độ đó bằng
`src/nondeterminism.py`: khoảng **0,83%** MAE trên bộ vàng và **0,70%** trên bộ xanh. Đây
là ngưỡng phân giải của cả nghiên cứu. Chênh lệch nhỏ hơn ngưỡng này không thể quy cho
can thiệp nào, và chúng tôi dùng chính nó để chấm lại các tuyên bố của mình.

**Không có rò rỉ nhãn.** Dữ liệu chia theo thời gian, không xáo trộn, mỗi mẫu được gán
vào tập theo thời điểm của nhãn. Khối hiệu chuẩn conformal nằm hẳn sau khối dừng sớm và
tách rời khỏi nó.

**Không lưu dự đoán và trọng số.** Tensor dự đoán và checkpoint mô hình bị loại khỏi repo
vì dung lượng. Chạy lại pipeline để sinh ra chúng.

**Kết quả âm được giữ nguyên.** Repo tái lập cả những kết quả không ủng hộ giả thuyết ban
đầu.

## Vì sao có repo này

Dự án khởi đầu từ một pipeline có sẵn. Bước đầu tiên là kiểm toán nó bằng
`src/audit_original.py`, và kết quả là:

```
corr(nhu cầu, historical_trip_count) = 1.0000000000
R² của hồi quy tuyến tính            = 1.0
```

Đặc trưng đầu vào **chính là nhãn**. Mô hình đang dự đoán thứ nó đã được cho biết, nên
toàn bộ số liệu của phiên bản cũ vô nghĩa và pipeline được xây lại từ đầu.

Script kiểm toán được giữ lại làm bằng chứng. Nó đọc các file kết quả của pipeline cũ qua
biến môi trường `GAT_ORIG`, và những file đó không nằm trong repo này.

## Công cụ cho bản thảo

Bốn script dùng để chuyển đổi và rà soát bản thảo: `markdown_to_latex.py`,
`export_paper_docx.py`, `audit_manuscript.py` và `update_robustness_manuscript.py`. Chúng
cần file nguồn của bản thảo, vốn không có trong repo, và không cần cho việc tái lập kết
quả.

## Giấy phép

Mã nguồn phát hành theo giấy phép MIT.

Dữ liệu **không** thuộc phạm vi giấy phép này. Mỗi bộ dữ liệu giữ điều khoản gốc của nó
(NYC TLC, Porto, Olist, Amazon, Meituan, Cainiao-LaDe). Hãy đọc điều khoản của từng bên
trước khi dùng. Riêng dữ liệu Meituan yêu cầu ghi lời cảm ơn, mô tả trong
`src/fetch_meituan_delivery.py`.

## Trích dẫn

```bibtex
@article{phamxuan2026forecastvalue,
  title   = {Forecast Value Is a Property of the Decision: Calibrated Quantile Demand
             Forecasting for Urban Last-Mile Capacity and Routing},
  author  = {Pham Xuan, Kien and Trinh Ngoc Minh, Nhat and Nguyen Thi My, Linh},
  journal = {International Journal of Analysis and Applications},
  year    = {2026},
  note    = {In press}
}
```

Tác giả liên hệ: Linh Nguyen Thi My, linh.ntm@vlu.edu.vn
