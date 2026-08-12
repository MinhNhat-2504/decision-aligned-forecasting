"""
Xay dung DO THI DA QUAN HE giua cac taxi zone.

Ban goc chi co 1 loai canh: k-NN theo khoang cach duong chim bay, va mang duong OSM
hoan toan khong tham gia vao cau truc do thi (chi dung o buoc ve tuyen).
O day ta tao 3 quan he bo sung cho nhau:

  R1 geo  : k-NN theo khoang cach haversine giua centroid  -> lan can khong gian
  R2 road : k-NN theo khoang cach NGAN NHAT TREN MANG DUONG OSM -> lan can kha thi khi di chuyen
            (hai zone sat nhau qua song co the rat xa nhau tren duong bo)
  R3 flow : k lien ket co luu luong OD lon nhat, CHI tinh tu khoang thoi gian TRAIN
            -> lan can theo hanh vi di chuyen, khong phai theo dia ly

Moi quan he duoc luu duoi dang edge_index [2, E] (huong: source -> target).
"""
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ZIP_PATH, OSM_MEMBER, PROC_DIR, K_GEO, K_ROAD, K_FLOW

R_EARTH = 6371.0


def haversine_matrix(lat, lon):
    """Ma tran khoang cach haversine [N, N] (km)."""
    la = np.radians(lat)[:, None]
    lo = np.radians(lon)[:, None]
    dlat = la - la.T
    dlon = lo - lo.T
    a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlon / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def knn_edges(dist, k, exclude_inf=True):
    """Lay k lang gieng gan nhat cho moi node tu ma tran khoang cach."""
    N = dist.shape[0]
    d = dist.copy().astype(np.float64)
    np.fill_diagonal(d, np.inf)
    src, dst, w = [], [], []
    for i in range(N):
        order = np.argsort(d[i])
        cnt = 0
        for j in order:
            if cnt >= k:
                break
            if not np.isfinite(d[i, j]) and exclude_inf:
                break
            src.append(i)
            dst.append(int(j))
            w.append(float(d[i, j]))
            cnt += 1
    return np.array(src), np.array(dst), np.array(w)


def symmetrize(src, dst):
    """Bien do thi thanh vo huong (them canh nguoc) va bo canh trung."""
    pairs = set()
    for s, t in zip(src, dst):
        pairs.add((int(s), int(t)))
        pairs.add((int(t), int(s)))
    arr = np.array(sorted(pairs), dtype=np.int64)
    return arr[:, 0], arr[:, 1]


def build_road_distance(zones):
    """
    Dung mang duong OSM -> khoang cach ngan nhat theo duong bo giua cac zone.
    Moi zone duoc neo vao node duong gan centroid nhat.
    """
    print("  doc OSM street network ...", flush=True)
    zf = zipfile.ZipFile(ZIP_PATH)
    df = pd.read_csv(zf.open(OSM_MEMBER), usecols=["geometry", "length"])

    # Lay 2 dau mut cua moi LINESTRING lam canh cua do thi duong
    import re
    coord_re = re.compile(r"(-?\d+\.?\d*)\s+(-?\d+\.?\d*)")
    node_id, edges = {}, []
    for geom, length in zip(df["geometry"].values, df["length"].values):
        if not isinstance(geom, str):
            continue
        ms = coord_re.findall(geom)
        if len(ms) < 2:
            continue
        u = (round(float(ms[0][0]), 6), round(float(ms[0][1]), 6))
        v = (round(float(ms[-1][0]), 6), round(float(ms[-1][1]), 6))
        if u == v:
            continue
        for p in (u, v):
            if p not in node_id:
                node_id[p] = len(node_id)
        L = float(length) if np.isfinite(length) and length > 0 else 1.0
        edges.append((node_id[u], node_id[v], L))

    M = len(node_id)
    print(f"  road graph: {M:,} nodes, {len(edges):,} segments", flush=True)

    rows = np.array([e[0] for e in edges] + [e[1] for e in edges])
    cols = np.array([e[1] for e in edges] + [e[0] for e in edges])
    vals = np.array([e[2] for e in edges] * 2, dtype=np.float64)
    A = csr_matrix((vals, (rows, cols)), shape=(M, M))

    # Neo centroid zone vao node duong gan nhat
    coords = np.array(sorted(node_id.keys(), key=lambda p: node_id[p]))  # (lon, lat)
    tree = cKDTree(coords)
    _, anchor = tree.query(np.c_[zones["lon"].values, zones["lat"].values])
    anchor = np.asarray(anchor, dtype=np.int64)
    print(f"  anchors: {len(np.unique(anchor))} node duong khac nhau", flush=True)

    t0 = time.time()
    D = dijkstra(A, directed=False, indices=anchor)   # [N, M] met
    Dz = D[:, anchor] / 1000.0                        # [N, N] km theo duong bo
    print(f"  dijkstra {time.time() - t0:.1f}s | vo cung: "
          f"{int(np.isinf(Dz).sum())}/{Dz.size}", flush=True)
    return Dz


def build():
    zones = pd.read_csv(os.path.join(PROC_DIR, "zones.csv"))
    N = len(zones)
    d = np.load(os.path.join(PROC_DIR, "demand_hourly.npz"))
    od = d["od_train"].astype(np.float64)

    print(f"Zones: {N}")

    # --- R1: geo ---
    print("\n[R1] geo k-NN (haversine)")
    Dgeo = haversine_matrix(zones["lat"].values, zones["lon"].values)
    s, t, _ = knn_edges(Dgeo, K_GEO)
    geo_s, geo_t = symmetrize(s, t)
    print(f"  edges: {len(geo_s)}")

    # --- R2: road ---
    print("\n[R2] road k-NN (shortest path tren OSM)")
    Droad = build_road_distance(zones)
    s, t, _ = knn_edges(Droad, K_ROAD)
    road_s, road_t = symmetrize(s, t)
    print(f"  edges: {len(road_s)}")

    # Do lech giua lang gieng dia ly va lang gieng duong bo
    set_geo = set(zip(geo_s.tolist(), geo_t.tolist()))
    set_road = set(zip(road_s.tolist(), road_t.tolist()))
    jac = len(set_geo & set_road) / len(set_geo | set_road)
    print(f"  Jaccard(geo, road) = {jac:.3f}  -> hai quan he khac nhau ro ret")

    # --- R3: OD flow (chi tu train) ---
    print("\n[R3] OD-flow k-NN (chi tu khoang TRAIN)")
    W = od + od.T
    np.fill_diagonal(W, 0)
    s, t = [], []
    for i in range(N):
        order = np.argsort(-W[i])[:K_FLOW]
        for j in order:
            if W[i, j] > 0:
                s.append(i)
                t.append(int(j))
    flow_s, flow_t = symmetrize(np.array(s), np.array(t))
    print(f"  edges: {len(flow_s)}")
    set_flow = set(zip(flow_s.tolist(), flow_t.tolist()))
    print(f"  Jaccard(geo, flow) = "
          f"{len(set_geo & set_flow) / len(set_geo | set_flow):.3f}")

    np.savez_compressed(
        os.path.join(PROC_DIR, "graphs.npz"),
        geo=np.stack([geo_s, geo_t]),
        road=np.stack([road_s, road_t]),
        flow=np.stack([flow_s, flow_t]),
        dist_geo=Dgeo.astype(np.float32),
        dist_road=np.where(np.isinf(Droad), 1e6, Droad).astype(np.float32),
    )
    print(f"\nSaved -> {os.path.join(PROC_DIR, 'graphs.npz')}")


if __name__ == "__main__":
    build()
