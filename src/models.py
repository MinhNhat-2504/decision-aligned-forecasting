"""
Cac mo hinh du bao nhu cau giao hang theo khong-thoi gian.

Bao gom:
  - Lop GAT DUNG CHUAN (Velickovic et al. 2018): multi-head, co self-loop,
    softmax vector hoa bang scatter (ban goc: 1 head, khong self-loop, vong for
    tren tung canh trong Python).
  - GCN / GraphSAGE de so sanh cong bang cung mot khung.
  - ST-MRGAT (de xuat): ma hoa thoi gian bang TCN + GAT DA QUAN HE hop nhat bang
    attention muc quan he + dau ra QUANTILE (pinball loss) cho khoang du bao.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- scatter utils
def scatter_softmax(logits, index, N):
    """Softmax theo nhom node dich. logits [B, E, H], index [E] -> [B, E, H]."""
    B, E, H = logits.shape
    idx = index.view(1, E, 1).expand(B, E, H)
    mx = torch.full((B, N, H), float("-inf"), device=logits.device, dtype=logits.dtype)
    mx = mx.scatter_reduce(1, idx, logits, reduce="amax", include_self=True)
    mx = torch.nan_to_num(mx, neginf=0.0)
    ex = (logits - mx.gather(1, idx)).exp()
    den = torch.zeros(B, N, H, device=logits.device, dtype=logits.dtype)
    den = den.scatter_add(1, idx, ex)
    return ex / (den.gather(1, idx) + 1e-16)


def scatter_sum(src, index, N):
    """Tong hop theo node dich. src [B, E, H, D], index [E] -> [B, N, H, D]."""
    B, E, H, D = src.shape
    idx = index.view(1, E, 1, 1).expand(B, E, H, D)
    out = torch.zeros(B, N, H, D, device=src.device, dtype=src.dtype)
    return out.scatter_add(1, idx, src)


def add_self_loops(edge_index, N):
    loop = torch.arange(N, device=edge_index.device)
    loop = torch.stack([loop, loop])
    return torch.cat([edge_index, loop], dim=1)


# ---------------------------------------------------------------- graph layers
class GATLayer(nn.Module):
    """Graph Attention Layer chuan: multi-head + self-loop + LeakyReLU(0.2)."""

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1, concat=True):
        super().__init__()
        self.h, self.d, self.concat = heads, out_dim // heads if concat else out_dim, concat
        self.W = nn.Linear(in_dim, self.h * self.d, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, 1, self.h, self.d))
        self.a_dst = nn.Parameter(torch.empty(1, 1, self.h, self.d))
        self.bias = nn.Parameter(torch.zeros(self.h * self.d if concat else self.d))
        self.dropout = dropout
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, x, edge_index, return_att=False):
        # x [B, N, F] ; edge_index [2, E] (source -> target)
        B, N, _ = x.shape
        ei = add_self_loops(edge_index, N)
        src, dst = ei[0], ei[1]

        h = self.W(x).view(B, N, self.h, self.d)                 # [B,N,H,D]
        a_s = (h * self.a_src).sum(-1)                            # [B,N,H]
        a_d = (h * self.a_dst).sum(-1)
        logits = F.leaky_relu(a_s[:, src] + a_d[:, dst], 0.2)     # [B,E,H]
        alpha = scatter_softmax(logits, dst, N)
        alpha_d = F.dropout(alpha, self.dropout, self.training)

        msg = h[:, src] * alpha_d.unsqueeze(-1)                   # [B,E,H,D]
        out = scatter_sum(msg, dst, N)                            # [B,N,H,D]
        out = out.reshape(B, N, -1) if self.concat else out.mean(2)
        out = out + self.bias
        return (out, alpha) if return_att else (out, None)


class GCNLayer(nn.Module):
    """GCN (Kipf & Welling 2017) voi chuan hoa doi xung D^-1/2 A D^-1/2."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, return_att=False):
        B, N, _ = x.shape
        ei = add_self_loops(edge_index, N)
        src, dst = ei[0], ei[1]
        deg = torch.zeros(N, device=x.device).index_add_(
            0, dst, torch.ones(dst.numel(), device=x.device))
        dinv = deg.clamp(min=1).pow(-0.5)
        w = (dinv[src] * dinv[dst]).view(1, -1, 1, 1)
        h = self.lin(x).unsqueeze(2)                              # [B,N,1,F]
        msg = h[:, src] * w
        out = scatter_sum(msg, dst, N).squeeze(2)
        return out, None


class SAGELayer(nn.Module):
    """GraphSAGE mean-aggregator (Hamilton et al. 2017)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, return_att=False):
        B, N, _ = x.shape
        src, dst = edge_index[0], edge_index[1]
        deg = torch.zeros(N, device=x.device).index_add_(
            0, dst, torch.ones(dst.numel(), device=x.device)).clamp(min=1)
        msg = x[:, src].unsqueeze(2)
        agg = scatter_sum(msg, dst, N).squeeze(2) / deg.view(1, N, 1)
        return self.lin_self(x) + self.lin_neigh(agg), None


# ---------------------------------------------------------------- temporal
class TemporalConv(nn.Module):
    """TCN nhan qua co dilation: ma hoa cua so lich su L gio thanh 1 vector/node."""

    def __init__(self, in_ch, hid, layers=3, k=3):
        super().__init__()
        self.convs, self.dils = nn.ModuleList(), []
        c = in_ch
        for i in range(layers):
            d = 2 ** i
            self.convs.append(nn.Conv1d(c, hid, k, dilation=d))
            self.dils.append((k - 1) * d)
            c = hid
        self.norm = nn.LayerNorm(hid)

    def forward(self, x):
        # x [B, N, L, C] -> [B, N, hid]
        B, N, L, C = x.shape
        h = x.reshape(B * N, L, C).transpose(1, 2)
        for conv, pad in zip(self.convs, self.dils):
            h = F.pad(h, (pad, 0))
            h = F.relu(conv(h))
        h = h[..., -1]
        return self.norm(h).view(B, N, -1)


# ---------------------------------------------------------------- models
def monotonic_quantiles(out, n_h, n_q):
    """
    [B, N, n_h*n_q] -> [B, N, n_h, n_q] voi cac phan vi KHONG CAT NHAU.
    Phan vi thap nhat tu do; cac phan vi sau cong them luong softplus duong.
    Dung chung cho moi backbone, khong rieng cho ST-MRGAT.
    """
    B, N, _ = out.shape
    out = out.view(B, N, n_h, n_q)
    if n_q == 1:
        return out
    lo = out[..., :1]
    inc = F.softplus(out[..., 1:])
    return torch.cat([lo, lo + torch.cumsum(inc, dim=-1)], dim=-1)


class GraphForecaster(nn.Module):
    """Khung chung: TCN thoi gian + 2 lop do thi (1 quan he) + dau ra quantile."""

    LAYERS = {"gat": GATLayer, "gcn": GCNLayer, "sage": SAGELayer}

    def __init__(self, in_ch, static_dim, hidden, n_out, kind="gat",
                 heads=4, dropout=0.1):
        super().__init__()
        self.tcn = TemporalConv(in_ch, hidden)
        self.static = nn.Linear(static_dim, hidden) if static_dim else None
        L = self.LAYERS[kind]
        mk = (lambda i, o: L(i, o, heads=heads, dropout=dropout)) if kind == "gat" \
            else (lambda i, o: L(i, o))
        self.g1, self.g2 = mk(hidden, hidden), mk(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                                  nn.Linear(hidden, n_out))

    def forward(self, x, static, edge_index, return_att=False):
        h = self.tcn(x)
        if self.static is not None:
            h = h + self.static(static).unsqueeze(0)
        skip = h
        h1, a1 = self.g1(h, edge_index, return_att)
        h = self.drop(F.elu(h1))
        h2, a2 = self.g2(h, edge_index, return_att)
        h = F.elu(h2)
        out = self.head(torch.cat([h, skip], -1))
        return (out, (a1, a2)) if return_att else (out, None)


class MRGATLayer(nn.Module):
    """
    GAT DA QUAN HE: chay 1 GAT rieng cho moi quan he (geo / road / flow),
    roi hop nhat bang attention muc quan he hoc duoc theo tung node.
    """

    def __init__(self, in_dim, out_dim, n_rel, heads=4, dropout=0.1):
        super().__init__()
        self.gats = nn.ModuleList(
            [GATLayer(in_dim, out_dim, heads, dropout) for _ in range(n_rel)])
        self.rel_score = nn.Linear(out_dim, 1)

    def forward(self, x, edge_list, return_att=False):
        outs, atts = [], []
        for gat, ei in zip(self.gats, edge_list):
            o, a = gat(x, ei, return_att)
            outs.append(o)
            atts.append(a)
        H = torch.stack(outs, dim=2)                       # [B,N,R,F]
        w = torch.softmax(self.rel_score(H), dim=2)        # [B,N,R,1]
        return (H * w).sum(2), (atts, w)


class STMRGAT(nn.Module):
    """
    Mo hinh de xuat: Spatio-Temporal Multi-Relational GAT + du bao quantile.

    - TCN ma hoa chuoi L gio lich su cua tung zone.
    - 2 lop MRGAT hop nhat 3 quan he khong gian bang attention muc quan he.
    - Dau ra: n_horizon x n_quantile -> co khoang du bao, khong chi 1 diem.
    """

    def __init__(self, in_ch, static_dim, hidden, n_horizon, n_quantile,
                 n_rel=3, heads=4, dropout=0.1):
        super().__init__()
        self.nh, self.nq = n_horizon, n_quantile
        self.tcn = TemporalConv(in_ch, hidden)
        self.static = nn.Linear(static_dim, hidden) if static_dim else None
        self.g1 = MRGATLayer(hidden, hidden, n_rel, heads, dropout)
        self.g2 = MRGATLayer(hidden, hidden, n_rel, heads, dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                                  nn.Linear(hidden, n_horizon * n_quantile))

    def forward(self, x, static, edge_list, return_att=False):
        h = self.tcn(x)
        if self.static is not None:
            h = h + self.static(static).unsqueeze(0)
        skip = h
        h1, a1 = self.g1(h, edge_list, return_att)
        h = self.drop(F.elu(h1))
        h2, a2 = self.g2(h, edge_list, return_att)
        h = F.elu(h2)
        out = monotonic_quantiles(self.head(torch.cat([h, skip], -1)),
                                  self.nh, self.nq)
        return (out, (a1, a2)) if return_att else (out, None)


class MLPForecaster(nn.Module):
    """Baseline KHONG dung do thi: MLP tren cua so lich su + dac trung tinh."""

    def __init__(self, in_ch, seq_len, static_dim, hidden, n_out, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch * seq_len + static_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_out))

    def forward(self, x, static, edge_index=None, return_att=False):
        B, N, L, C = x.shape
        f = x.reshape(B, N, L * C)
        f = torch.cat([f, static.unsqueeze(0).expand(B, -1, -1)], -1)
        return self.net(f), None


class LSTMForecaster(nn.Module):
    """Baseline KHONG dung do thi: LSTM tren chuoi thoi gian tung zone."""

    def __init__(self, in_ch, static_dim, hidden, n_out, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, hidden, num_layers=2, batch_first=True,
                            dropout=dropout)
        self.static = nn.Linear(static_dim, hidden) if static_dim else None
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x, static, edge_index=None, return_att=False):
        B, N, L, C = x.shape
        h, _ = self.lstm(x.reshape(B * N, L, C))
        h = h[:, -1].view(B, N, -1)
        if self.static is not None:
            h = h + self.static(static).unsqueeze(0)
        return self.head(h), None


# ---------------------------------------------------------------- losses
def pinball_loss(pred, target, quantiles):
    """
    pred   [B, N, H, Q]
    target [B, N, H]
    """
    t = target.unsqueeze(-1)
    q = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype)
    e = t - pred
    return torch.maximum(q * e, (q - 1) * e).mean()


# ================================================================ ST-GNN baselines
# Hai baseline chuyen dung cho du bao khong-thoi gian, cai dat lai trong cung khung
# (cung dau vao, cung nen mua vu, cung vong huan luyen) de so sanh cong bang.

def dense_transition(edge_index, N, device, dtype):
    """Ma tran chuyen buoc ngau nhien chuan hoa theo hang, co self-loop."""
    A = torch.zeros(N, N, device=device, dtype=dtype)
    A[edge_index[0], edge_index[1]] = 1.0
    A = A + torch.eye(N, device=device, dtype=dtype)
    return A / A.sum(1, keepdim=True).clamp(min=1)


def diffusion(x, supports, K):
    """Tich chap khuech tan: gom [x, Sx, S^2x, ...] cho moi support."""
    out = [x]
    for S in supports:
        xk = x
        for _ in range(K):
            xk = torch.einsum("ij,bjf->bif", S, xk)
            out.append(xk)
    return torch.cat(out, dim=-1)


class DCGRUCell(nn.Module):
    """O GRU trong do moi phep nhan ma tran duoc thay bang tich chap khuech tan."""

    def __init__(self, in_dim, hid, n_support, K=2):
        super().__init__()
        m = 1 + n_support * K
        self.hid, self.K = hid, K
        self.gates = nn.Linear((in_dim + hid) * m, 2 * hid)
        self.cand = nn.Linear((in_dim + hid) * m, hid)

    def forward(self, x, h, supports):
        ru = torch.sigmoid(self.gates(diffusion(torch.cat([x, h], -1),
                                                supports, self.K)))
        r, u = ru.chunk(2, dim=-1)
        c = torch.tanh(self.cand(diffusion(torch.cat([x, r * h], -1),
                                           supports, self.K)))
        return u * h + (1 - u) * c


class DCRNN(nn.Module):
    """
    Diffusion Convolutional Recurrent Neural Network (Li et al., ICLR 2018).
    Ban cai dat o day la phan ENCODER; bo giai ma co scheduled sampling duoc
    thay bang mot dau tuyen tinh - can ghi ro trong bai bao.
    """

    def __init__(self, in_ch, static_dim, hidden, n_out, K=2, layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DCGRUCell(in_ch if i == 0 else hidden, hidden, 2, K)
            for i in range(layers)])
        self.hid = hidden
        self.static = nn.Linear(static_dim, hidden) if static_dim else None
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x, static, edge_index, return_att=False):
        B, N, L, C = x.shape
        P = dense_transition(edge_index, N, x.device, x.dtype)
        supports = [P, P.t()]
        hs = [torch.zeros(B, N, self.hid, device=x.device, dtype=x.dtype)
              for _ in self.layers]
        for t in range(L):
            inp = x[:, :, t]
            for i, cell in enumerate(self.layers):
                hs[i] = cell(inp, hs[i], supports)
                inp = hs[i]
        h = hs[-1]
        if self.static is not None:
            h = h + self.static(static).unsqueeze(0)
        return self.head(self.drop(h)), None


class GWNetBlock(nn.Module):
    """Mot khoi Graph WaveNet: TCN nhan qua co cong + tich chap do thi."""

    def __init__(self, ch, n_support, K=2, dilation=1, kernel=2):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.filt = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.gate = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.gconv = nn.Linear(ch * (1 + n_support * K), ch)
        self.K = K
        self.norm = nn.LayerNorm(ch)

    def forward(self, h, supports, N):
        # h: [B*N, ch, L]
        z = F.pad(h, (self.pad, 0))
        z = torch.tanh(self.filt(z)) * torch.sigmoid(self.gate(z))
        BN, ch, L = z.shape
        B = BN // N
        s = z.permute(0, 2, 1).reshape(B, N, L * ch)          # gom truc thoi gian
        s = s.view(B, N, L, ch).permute(0, 2, 1, 3).reshape(B * L, N, ch)
        s = self.gconv(diffusion(s, supports, self.K))
        s = s.view(B, L, N, ch).permute(0, 2, 1, 3).reshape(B * N, L, ch)
        s = self.norm(s).permute(0, 2, 1)
        return h + s, s


class GraphWaveNet(nn.Module):
    """
    Graph WaveNet (Wu et al., IJCAI 2019).

    Dac trung nhan dang cua mo hinh la MA TRAN KE THICH NGHI hoc duoc
    A_adp = softmax(relu(E1 E2^T)), tu phat hien quan he giua cac nut ma do thi
    cho truoc khong co. Doi voi bai bao nay no con la mot phep thu doc lap:
    neu quan he HOC DUOC cung khong vuot duoc do thi dia ly don gian thi ket luan
    "cau truc do thi bo sung khong mua duoc gi" khong phai do ta chon sai quan he.
    """

    def __init__(self, in_ch, static_dim, hidden, n_out, N=263, emb=10,
                 K=2, dropout=0.1, n_quantile=1):
        super().__init__()
        self.nh, self.nq = n_out, n_quantile
        self.start = nn.Conv1d(in_ch, hidden, 1)
        self.E1 = nn.Parameter(torch.randn(N, emb) * 0.1)
        self.E2 = nn.Parameter(torch.randn(N, emb) * 0.1)
        self.blocks = nn.ModuleList([
            GWNetBlock(hidden, 3, K, dilation=d) for d in (1, 2, 4)])
        self.static = nn.Linear(static_dim, hidden) if static_dim else None
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, n_out * n_quantile))

    def forward(self, x, static, edge_index, return_att=False):
        B, N, L, C = x.shape
        P = dense_transition(edge_index, N, x.device, x.dtype)
        A_adp = torch.softmax(F.relu(self.E1 @ self.E2.t()), dim=1)
        supports = [P, P.t(), A_adp]

        h = self.start(x.reshape(B * N, L, C).transpose(1, 2))
        skips = 0
        for blk in self.blocks:
            h, s = blk(h, supports, N)
            skips = skips + s[..., -1:]
        z = skips.squeeze(-1).view(B, N, -1)
        if self.static is not None:
            z = z + self.static(static).unsqueeze(0)
        out = self.head(self.drop(F.relu(z)))
        if self.nq > 1:
            out = monotonic_quantiles(out, self.nh, self.nq)
        return out, None
