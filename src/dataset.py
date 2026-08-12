"""
Tao cua so truot + chia tap theo THOI GIAN + chuan hoa CHI TU TAP TRAIN.

Ba khoi thoi gian lien tuc theo NHAN (target):
    TRAIN  2016-01-01 .. 2016-02-29
    VAL    2016-03-01 .. 2016-03-15
    TEST   2016-03-16 .. 2016-03-31

Dau vao duoc phep nhin NGUOC ve qua khu qua bien khoi (vi du mau test dung quan sat
cua tuan truoc do): day la nhan qua va dung voi tinh huong trien khai that.
Ro ri chi xay ra neu dung thong tin TUONG LAI, dieu khong ton tai o day.

Dau vao da thanh phan (kieu ASTGCN): nhu cau taxi co chu ky ngay va tuan rat manh,
nen moi buoc thoi gian mang 3 kenh nhu cau:
    z(t)      : gan
    z(t-24)   : cung gio hom truoc
    z(t-168)  : cung gio tuan truoc
cong 5 dac trung lich -> C = 8.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (PROC_DIR, SEQ_LEN, HORIZONS, TARGET, T_START, T_END,
                    TRAIN_END, VAL_END, DEMAND_FILE, GRAPH_FILE)

LAG_D, LAG_W = 24, 168


def _time_features(ts):
    hod = ts.hour.values.astype(np.float32)
    dow = ts.dayofweek.values.astype(np.float32)
    return np.stack([
        np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        (dow >= 5).astype(np.float32),
    ], axis=1)


def _shift(a, k):
    """Dich chuoi ve qua khu k buoc; k buoc dau lay gia tri dau tien."""
    out = np.empty_like(a)
    out[k:] = a[:-k]
    out[:k] = a[0]
    return out


class STData:
    def __init__(self, device="cpu"):
        d = np.load(os.path.join(PROC_DIR, DEMAND_FILE))
        self.zones = pd.read_csv(os.path.join(PROC_DIR, "zones.csv"))
        y = d[TARGET].astype(np.float32)
        # T_END may be shortened for a rolling-origin backtest.  The source npz
        # contains the full quarter; slicing only removes future observations and
        # therefore cannot leak information into an earlier evaluation origin.
        requested_ts = pd.date_range(T_START, T_END, freq="h")
        if len(requested_ts) > len(y):
            raise ValueError(f"Requested {len(requested_ts)} hours but data has {len(y)}")
        y = y[:len(requested_ts)]
        self.raw = y
        T, N = y.shape
        self.T, self.N = T, N

        ts = requested_ts
        assert len(ts) == T, (len(ts), T)
        self.ts = ts

        i_tr = int(np.searchsorted(ts, pd.Timestamp(TRAIN_END))) + 1
        i_va = int(np.searchsorted(ts, pd.Timestamp(VAL_END))) + 1
        self.bounds = (0, i_tr, i_va, T)

        ly = np.log1p(y)
        self.mu = float(ly[:i_tr].mean())
        self.sd = float(ly[:i_tr].std() + 1e-8)
        self.z = ((ly - self.mu) / self.sd).astype(np.float32)

        self._tf = _time_features(ts)
        self._N, self._ts = N, ts

        zs = self.zones
        bor = pd.get_dummies(zs["borough"]).values.astype(np.float32)
        area = np.log1p(zs["area_km2"].values).astype(np.float32)[:, None]
        lat = ((zs["lat"].values - zs["lat"].mean()) / zs["lat"].std()).astype(np.float32)[:, None]
        lon = ((zs["lon"].values - zs["lon"].mean()) / zs["lon"].std()).astype(np.float32)[:, None]
        self.static = np.concatenate([area, lat, lon, bor], axis=1).astype(np.float32)
        self.S = self.static.shape[1]

        # --- Nen mua vu: trung binh lich su theo (gio-trong-tuan, zone) ---
        # Uoc luong CHI tren train. Cac mo hinh neural du bao PHAN DU tren nen nay,
        # nen khong phai hoc lai chu ky ngay/tuan von rat manh cua nhu cau do thi.
        self.how = (ts.dayofweek.values * 24 + ts.hour.values).astype(np.int64)
        tab = np.zeros((168, N), dtype=np.float64)
        cnt = np.zeros(168, dtype=np.float64)
        for t in range(i_tr):
            tab[self.how[t]] += y[t]
            cnt[self.how[t]] += 1
        self.ha_raw = (tab / np.maximum(cnt, 1)[:, None]).astype(np.float32)
        self.ha_z = ((np.log1p(self.ha_raw) - self.mu) / self.sd).astype(np.float32)

        # Nen mua vu trai theo truc thoi gian, va PHAN DU quan sat duoc.
        # Kenh phan du la mau chot: mo hinh du bao phan du nen phai NHIN THAY
        # phan du gan nhat, thay vi phai tu tai tao chu ky 168 gio trong noi bo.
        ha_t = self.ha_z[self.how]                                   # [T, N]
        resid = (self.z - ha_t).astype(np.float32)
        tf = np.repeat(self._tf[:, None, :], N, axis=1)
        self.x_all = np.concatenate([
            self.z[..., None],
            resid[..., None],
            _shift(resid, LAG_D)[..., None],
            _shift(resid, LAG_W)[..., None],
            ha_t[..., None],
            tf,
        ], axis=2).astype(np.float32)
        self.C = self.x_all.shape[2]

        self.max_h = max(HORIZONS)
        self.idx = {k: self._windows(lo, hi) for k, (lo, hi) in
                    dict(train=(0, i_tr), val=(i_tr, i_va), test=(i_va, T)).items()}

        # Tach VAL thanh hai phan roi nhau theo thoi gian:
        #   val_es  (60% dau)  -> chon epoch dung bo
        #   val_cal (40% cuoi) -> hieu chuan conformal
        # CQR doi tap hieu chuan KHONG tham gia vao viec chon mo hinh; neu dung
        # chung mot tap cho ca hai thi bao dam do phu khong con chat che.
        v = self.idx["val"]
        cut = int(len(v) * 0.6)
        self.idx["val_es"], self.idx["val_cal"] = v[:cut], v[cut:]
        self._cache = {}
        self.device = device

    def _windows(self, lo, hi):
        """
        s = chi so bat dau cua so dau vao.
        Rang buoc: du lich su (s >= LAG_W) va TOAN BO nhan roi vao [lo, hi).
        """
        s = np.arange(LAG_W, self.T)
        first_t = s + SEQ_LEN + HORIZONS[0] - 1
        last_t = s + SEQ_LEN + self.max_h - 1
        return s[(first_t >= lo) & (last_t < hi)]

    def get(self, split, ids=None):
        """Tra ve (x, y_z, y_raw, base_z) voi base_z la nen mua vu tai gio can du bao."""
        if ids is None:
            if split in self._cache:
                return self._cache[split]
            out = self._build(self.idx[split])
            self._cache[split] = out
            return out
        return self._build(ids)

    def _build(self, start):
        L, H = SEQ_LEN, HORIZONS
        xs = np.stack([self.x_all[s:s + L] for s in start]).transpose(0, 2, 1, 3)
        tgt_t = np.array([[s + L + h - 1 for h in H] for s in start])
        yz = self.z[tgt_t].transpose(0, 2, 1)
        yr = self.raw[tgt_t].transpose(0, 2, 1)
        base = self.ha_z[self.how[tgt_t]].transpose(0, 2, 1)
        f32 = lambda a: torch.from_numpy(np.ascontiguousarray(a.astype(np.float32)))
        return f32(xs), f32(yz), f32(yr), f32(base)

    def target_times(self, split):
        return np.array([[s + SEQ_LEN + h - 1 for h in HORIZONS]
                         for s in self.idx[split]])

    def inverse(self, z):
        return np.expm1(np.clip(z, -50, 50) * self.sd + self.mu)

    def summary(self):
        b = self.bounds
        print(f"T={self.T} N={self.N} C={self.C} S={self.S} "
              f"(kenh: z(t), z(t-24), z(t-168) + 5 dac trung lich)")
        for k, i in (("train", 1), ("val", 2), ("test", 3)):
            tt = self.target_times(k)
            print(f"  {k:5s} windows={len(self.idx[k]):5d} "
                  f"nhan {self.ts[tt.min()]} -> {self.ts[tt.max()]}")
        print(f"  scaler(log1p,z) mu={self.mu:.4f} sd={self.sd:.4f}  (chi tu train)")


def load_graphs(device="cpu"):
    g = np.load(os.path.join(PROC_DIR, GRAPH_FILE))
    return {k: torch.from_numpy(g[k].astype(np.int64)).to(device)
            for k in ("geo", "road", "flow")}


if __name__ == "__main__":
    d = STData()
    d.summary()
    x, yz, yr, base = d.get("test")
    print("test batch:", x.shape, yz.shape, yr.shape, base.shape)
