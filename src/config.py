"""Cau hinh trung tam cho pipeline ST-GAT last-mile delivery."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Duong dan toi kho du lieu goc. Dat qua bien moi truong de khong phu thuoc may:
#   export GAT_ZIP="/duong/dan/toi/GAT_LastMile_Delivery 3.zip"
ZIP_PATH = os.environ.get(
    "GAT_ZIP", r"d:/Project/AI Engineer/Paper1/GAT_LastMile_Delivery 3.zip")
ZIP_ROOT = "GAT_LastMile_Delivery"

# Chon bo du lieu qua bien moi truong: DATASET=yellow (mac dinh) | green
DATASET = os.environ.get("DATASET", "yellow")
_SUF = "" if DATASET == "yellow" else f"_{DATASET}"

PROC_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs" + _SUF)
FIG_DIR = os.path.join(BASE_DIR, "figures" + _SUF)
DEMAND_FILE = f"demand_hourly{_SUF}.npz"
GRAPH_FILE = f"graphs{_SUF}.npz"
for _d in (PROC_DIR, OUT_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)

# --- Nguon du lieu trong zip -------------------------------------------------
ZONES_MEMBER = f"{ZIP_ROOT}/data/raw/Taxi Zones/taxi_zones.csv"
OSM_MEMBER = f"{ZIP_ROOT}/data/raw/Street Network OSM/new_york_streets.csv"
TRIP_MEMBERS = [
    f"{ZIP_ROOT}/data/raw/Taxi Trip Data/yellow_tripdata_2016-01.csv",
    f"{ZIP_ROOT}/data/raw/Taxi Trip Data/yellow_tripdata_2016-02.csv",
    f"{ZIP_ROOT}/data/raw/Taxi Trip Data/yellow_tripdata_2016-03.csv",
]

# --- Mien thoi gian ----------------------------------------------------------
T_START = os.environ.get("T_START", "2016-01-01 00:00:00")
T_END = os.environ.get("T_END", "2016-03-31 23:00:00")

# Chia theo thoi gian (KHONG shuffle) - ranh gioi tinh theo gio
TRAIN_END = os.environ.get("TRAIN_END", "2016-02-29 23:00:00")   # 1440 gio
VAL_END = os.environ.get("VAL_END", "2016-03-15 23:00:00")       # 360 gio
# test: 2016-03-16 00:00 -> 2016-03-31 23:00 (384 gio)

# --- Bai toan du bao ---------------------------------------------------------
SEQ_LEN = 12          # so gio lich su lam dau vao
HORIZONS = [1, 3, 6]  # du bao t+1, t+3, t+6 gio
TARGET = "dropoff"    # proxy nhu cau giao hang: so luot tra khach den zone

# --- Do thi ------------------------------------------------------------------
K_GEO = 8             # k-NN theo khoang cach duong chim bay
K_ROAD = 8            # k-NN theo khoang cach tren mang duong OSM
K_FLOW = 8            # k lien ket OD-flow manh nhat (chi tinh tren tap TRAIN)

# --- Huan luyen --------------------------------------------------------------
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 64))
N_HEADS = int(os.environ.get("N_HEADS", 4))
DROPOUT = float(os.environ.get("DROPOUT", 0.1))
LR = float(os.environ.get("LR", 1e-3))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 1e-4))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", 150))
PATIENCE = int(os.environ.get("PATIENCE", 25))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
SEEDS = [0, 1, 2, 3, 4]
QUANTILES = [0.1, 0.5, 0.9]
