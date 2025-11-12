# ---------------------------------------------------------
# 1) 编码器：保留通道独立 Transformer，移除 FFT，用 WTSConv 做小波时空卷积
# 2) EEG Brain-region Feature Decomposer：按论文，真实“脑区即超边”的超图 + 结构编码
# 3) Visual Brain-region Feature Decomposer：按论文，RGC(GAT) + BFG，输入为上游 patch token
# 仅包含模型与损失，在文件末尾提供一个可直接运行的 random 测试用例
# ---------------------------------------------------------

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 需要 PyTorch Geometric
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GATConv, HypergraphConv
    from torch_geometric.utils import to_dense_batch
except Exception as e:
    raise ImportError("This module requires torch_geometric. Please install PyG. Error: {}".format(e))


# -----------------------------
# 通用小工具
# -----------------------------

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


# -----------------------------
# 1) 通道独立的时序 Transformer
# -----------------------------

class ChannelWiseTransformer(nn.Module):
    """
    对每个通道的时序序列独立地应用同一套 Transformer Encoder（共享参数）。
    输入: x [B, C, T]
    输出: y [B, C, T] （通过线性层从 d_model 压回标量）
    """
    def __init__(self, d_model: int = 128, nhead: int = 8, num_layers: int = 2,
                 dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj_in = nn.Linear(1, d_model)
        self.proj_out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        B, C, T = x.shape
        x_seq = x.reshape(B * C, T, 1)       # [B*C, T, 1]
        h = self.proj_in(x_seq)              # [B*C, T, d_model]
        h = self.encoder(h)                  # [B*C, T, d_model]
        y = self.proj_out(h)                 # [B*C, T, 1]
        y = y.reshape(B, C, T)               # [B, C, T]
        return y


# -----------------------------
# 2) WTSConv：小波时间卷积 + 空间卷积
# -----------------------------

def _make_wavelet_filter_1d(wave: str, k_size: int) -> torch.Tensor:
    """
    简化的小波核：高斯调制正弦（近似 Morlet）
    返回: [k_size] 一维核
    """
    t = torch.linspace(-2.5, 2.5, steps=k_size)
    if wave.lower() in ["morl", "morlet", "cmor"]:
        s = torch.exp(-t**2)
        w = torch.cos(5.0 * t) * s
    else:
        w = torch.exp(-t**2)
    w = w - w.mean()
    w = w / (w.abs().sum() + 1e-8)
    return w


class WTSConv(nn.Module):
    """
    WTSConv：先小波时间卷积（沿 T 维），残差加回；再平均池化与小波空间卷积（沿 C 维），残差加回。
    输入: x_ts [B, C, T] —— 由 Transformer 增强后的时序特征
    输出: x_out [B, C, T]
    """
    def __init__(self, channels: int = 63, t_kernel: int = 25,
                 wave: str = "morl", pool_t: int = 5):
        super().__init__()
        self.channels = channels
        self.t_kernel = t_kernel
        self.wave = wave
        self.pool_t = pool_t

        # 时间卷积：在 [B, 1, C, T] 上用 (1, t_kernel) 做 conv2d
        self.time_in = nn.Conv2d(1, 32, kernel_size=(1, t_kernel),
                                 padding=(0, t_kernel // 2), bias=False)
        self.time_out = nn.Conv2d(32, 1, kernel_size=1, bias=False)

        # 空间卷积：在池化后 [B, 1, C, T'] 上用 (C, 1) 做 conv2d
        self.space_in = nn.Conv2d(1, 32, kernel_size=(channels, 1),
                                  padding=(0, 0), bias=False)
        self.space_out = nn.Conv2d(32, 1, kernel_size=1, bias=False)

        # 波形原子（调制时间卷积核）
        w_time = _make_wavelet_filter_1d(wave, t_kernel)
        self.register_buffer("w_time_atom", w_time.view(1, 1, 1, -1))  # [1,1,1,k]

        self.bn_t = nn.BatchNorm2d(1)
        self.bn_s = nn.BatchNorm2d(1)
        self.act = nn.SiLU()

        self.res_proj_t = nn.Identity()
        self.res_proj_s = nn.Identity()

    def forward(self, x_ts: torch.Tensor) -> torch.Tensor:
        # x_ts: [B, C, T]
        B, C, T = x_ts.shape
        x = x_ts.unsqueeze(1)  # [B, 1, C, T]

        # ---- 时间小波卷积 ----
        k = self.time_in.kernel_size[1]
        w = self.time_in.weight                  # [32,1,1,k]
        w = w * self.w_time_atom                 # 调制为小波形状
        y = F.conv2d(x, w, padding=(0, k // 2))  # [B,32,C,T]
        y = self.time_out(y)                     # [B,1,C,T]
        y = self.bn_t(y)
        y = self.act(y)
        xr = self.res_proj_t(x)                  # [B,1,C,T]
        y = y + xr                               # 残差

        # ---- 时间池化 ----
        if self.pool_t > 1:
            y_pool = F.avg_pool2d(
                y, kernel_size=(1, self.pool_t), stride=(1, self.pool_t)
            )                                   # [B,1,C,Tp]
        else:
            y_pool = y

        # ---- 空间小波卷积 ----
        w_s = self.space_in.weight               # [32,1,C,1]
        ys = F.conv2d(y_pool, w_s)               # [B,32,1,Tp]
        ys = self.space_out(ys)                  # [B,1,1,Tp]
        ys = ys.expand(-1, -1, C, -1)            # [B,1,C,Tp]
        ys = self.bn_s(ys)
        ys = self.act(ys)
        yr = self.res_proj_s(y_pool)             # [B,1,C,Tp]
        ys = ys + yr                             # 残差

        # ---- 插值回原时间长度 ----
        if ys.shape[-1] != T:
            ys = F.interpolate(ys, size=(C, T), mode="bilinear", align_corners=False)

        return ys.squeeze(1)                     # [B, C, T]


# -----------------------------
# 3) EEG Brain-region Feature Decomposer (EBFD) —— 超图
# -----------------------------

EEG_63_CH_ORDER = [
    'Fp1','Fp2','AF7','AF3','AFz','AF4','AF8',
    'F7','F5','F3','F1','F2','F4','F6','F8',
    'FT9','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','FT10',
    'T7','C5','C3','C1','Cz','C2','C4','C6','T8',
    'TP9','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','TP10',
    'P7','P5','P3','P1','Pz','P2','P4','P6','P8',
    'PO7','PO3','POz','PO4','PO8','O1','Oz','O2'
]

BRAIN_REGIONS = {
    'Frontal': ['Fp1','Fp2','AF7','AF3','AFz','AF4','AF8',
                'F7','F5','F3','F1','F2','F4','F6','F8',
                'FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT7','FT8'],
    'Sensorimotor': ['C5','C3','C1','Cz','C2','C4','C6',
                     'FC5','FC3','FC1','FCz','FC2','FC4','FC6',
                     'CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP9','TP10'],
    'Temporal': ['T7','T8','FT9','FT7','FT8','FT10','TP7','TP8'],
    'Parietal': ['P7','P5','P3','P1','Pz','P2','P4','P6','P8',
                 'CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP7','TP8'],
    'Occipital': ['PO7','PO3','POz','PO4','PO8','O1','Oz','O2']
}
REGION_LIST = ['Frontal','Sensorimotor','Temporal','Parietal','Occipital']


def channel_indices_of(region: str) -> List[int]:
    idx = []
    name2idx = {n: i for i, n in enumerate(EEG_63_CH_ORDER)}
    for ch in BRAIN_REGIONS[region]:
        if ch in name2idx:
            idx.append(name2idx[ch])
    return idx


def build_hyperedge_index_for_regions(num_nodes: int = 63) -> Tuple[torch.Tensor, int]:
    """
    构建超图 incidence：第二行是超边 id (脑区)，第一行是节点 id (电极)
    """
    node_ids = []
    hedge_ids = []
    for h_id, region in enumerate(REGION_LIST):
        ch_idx = channel_indices_of(region)
        for v in ch_idx:
            node_ids.append(v)
            hedge_ids.append(h_id)
    edge_index = torch.tensor([node_ids, hedge_ids], dtype=torch.long)
    return edge_index, len(REGION_LIST)


def hypergraph_laplacian_eigvecs(num_nodes: int, num_edges: int,
                                 edge_index: torch.Tensor, k: int = 8,
                                 device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    超图拉普拉斯 L = I - Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}
    返回前 k 个特征向量 [N,k]
    """
    H = torch.zeros((num_nodes, num_edges), device=device)
    H[edge_index[0], edge_index[1]] = 1.0
    De = torch.diag(H.sum(dim=0))   # 超边度
    Dv = torch.diag(H.sum(dim=1))   # 节点度
    De_inv = torch.inverse(De + 1e-6 * torch.eye(num_edges, device=device))
    Dv_inv_sqrt = torch.inverse(torch.sqrt(Dv + 1e-6 * torch.eye(num_nodes, device=device)))
    L = torch.eye(num_nodes, device=device) - Dv_inv_sqrt @ H @ De_inv @ H.t() @ Dv_inv_sqrt
    eigvals, eigvecs = torch.linalg.eigh(L)
    k = min(k, eigvecs.shape[1])
    return eigvecs[:, :k]           # [N,k]


def default_2d_layout(num_nodes: int = 63, radius: float = 1.0,
                      device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    没有实际坐标时，使用圆形布局近似，方便计算接近中心性
    """
    theta = torch.linspace(0, 2*math.pi, steps=num_nodes+1,
                           device=device, dtype=torch.float32)[:-1]
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    return torch.stack([x, y], dim=-1)  # [N,2]


def closeness_centrality(coords: torch.Tensor) -> torch.Tensor:
    """
    coords: [N,2] or [N,3]，返回 [N,1] 近似接近中心性（曼哈顿距离）
    """
    N, D = coords.shape
    dists = torch.cdist(coords, coords, p=1)
    sums = dists.sum(dim=1, keepdim=True) + 1e-6
    closeness = 1.0 / sums
    closeness = (closeness - closeness.min()) / (closeness.max() - closeness.min() + 1e-6)
    return closeness


def uniqueness_from_incidence(num_nodes: int, num_edges: int,
                              edge_index: torch.Tensor) -> torch.Tensor:
    """
    唯一性：节点跨超边参与次数的反比；返回 [N,1]
    """
    H = torch.zeros((num_nodes, num_edges), device=edge_index.device)
    H[edge_index[0], edge_index[1]] = 1.0
    counts = H.sum(dim=1, keepdim=True)
    uniq = 1.0 / (counts + 1e-6)
    uniq = (uniq - uniq.min()) / (uniq.max() - uniq.min() + 1e-6)
    return uniq


class EBFD(nn.Module):
    """
    EEG Brain-region Feature Decomposer
    输入: eeg_node_feat [B,63,D] （每通道向量）
         electrode_coords [63,2/3] 可选
    输出: local_feats [B,5,out_dim], global_feat [B,out_dim]
    """
    def __init__(self, in_dim: int, hidden_dim: int = 256,
                 out_dim: int = 1024, k_pe: int = 8,
                 knn_k: int = 4, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.k_pe = k_pe
        self.knn_k = knn_k
        self.num_layers = num_layers
        self.dropout = dropout

        # 功能超图卷积堆叠
        self.hg_layers = nn.ModuleList([
            HypergraphConv(in_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        # 结构编码投影
        self.proj_cent = nn.Linear(1, hidden_dim)
        self.proj_uniq = nn.Linear(1, hidden_dim)
        self.proj_pe   = nn.Linear(k_pe, hidden_dim)

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.SiLU()

        # 局部/全局投影
        self.local_proj  = nn.Linear(hidden_dim, out_dim)
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, eeg_node_feat: torch.Tensor,
                electrode_coords: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = eeg_node_feat.device
        B, N, D = eeg_node_feat.shape
        assert N == 63, "当前实现假定 63 通道"

        # 超图 incidence（固定 5 个脑区超边）
        hedge_index, num_hedges = build_hyperedge_index_for_regions(num_nodes=N)
        hedge_index = hedge_index.to(device)

        # 结构信息（与 batch 无关）
        if electrode_coords is None:
            coords = default_2d_layout(num_nodes=N, device=device)  # [63,2]
        else:
            coords = electrode_coords.to(device).float()
            assert coords.shape[0] == N

        cent = closeness_centrality(coords)                         # [N,1]
        uniq = uniqueness_from_incidence(N, num_hedges, hedge_index)  # [N,1]
        pe   = hypergraph_laplacian_eigvecs(N, num_hedges, hedge_index,
                                            k=self.k_pe, device=device)  # [N,k]

        cent_emb = self.proj_cent(cent)    # [N,hidden]
        uniq_emb = self.proj_uniq(uniq)    # [N,hidden]
        pe_emb   = self.proj_pe(pe)        # [N,hidden]
        struct_mul = cent_emb * uniq_emb * pe_emb  # 逐元素乘法（符合论文）

        # 功能超图卷积
        x_list = []
        for b in range(B):
            x = eeg_node_feat[b]          # [N,D]
            for i, hg in enumerate(self.hg_layers):
                x = hg(x, hedge_index)    # [N,hidden]
                x = self.act(x)
                if i < len(self.hg_layers) - 1:
                    x = F.dropout(x, p=self.dropout, training=self.training)
            x_list.append(x)
        x_fun = torch.stack(x_list, dim=0)  # [B,N,hidden]

        # 结构融合（broadcast 到 batch）
        x_str = struct_mul.unsqueeze(0).expand(B, -1, -1)  # [B,N,hidden]
        x_all = x_fun + x_str
        x_all = self.bn(x_all.reshape(-1, x_all.shape[-1])).reshape(B, N, -1)
        x_all = self.act(x_all)

        # 局部：按脑区聚合
        local_list = []
        name2idx = {n: i for i, n in enumerate(EEG_63_CH_ORDER)}
        for region in REGION_LIST:
            idx = [name2idx[n] for n in BRAIN_REGIONS[region] if n in name2idx]
            idx_t = torch.tensor(idx, device=device, dtype=torch.long)
            region_feat = x_all.index_select(1, idx_t).mean(dim=1)  # [B,hidden]
            local_list.append(self.local_proj(region_feat))         # [B,out_dim]
        local_feats = torch.stack(local_list, dim=1)                # [B,5,out_dim]

        # 全局：节点平均
        global_feat = self.global_proj(x_all.mean(dim=1))           # [B,out_dim]
        return local_feats, global_feat


# -----------------------------
# 4) Visual Brain-region Feature Decomposer (VBFD)
# -----------------------------

class RGCBackbone(nn.Module):
    """
    Region-aware Graph Convolution backbone：GAT over merged (explicit + KNN) edges.
    """
    def __init__(self, dim: int, heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                GATConv(dim, dim // heads, heads=heads, concat=True, dropout=dropout)
            )
        self.act = nn.SiLU()
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for gat in self.layers:
            h = gat(h, edge_index)
            h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


def build_knn_edges(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    基于余弦相似度的 kNN 图
    """
    with torch.no_grad():
        x_n = l2_normalize(x, dim=-1)
        sim = x_n @ x_n.t()                 # [M,M]
        M = x.shape[0]
        sim.fill_diagonal_(-1e9)
        topk = sim.topk(k=k, dim=-1).indices
        row = torch.arange(M, device=x.device).unsqueeze(1).expand_as(topk).reshape(-1)
        col = topk.reshape(-1)
        edge_index = torch.stack([row, col], dim=0)
    return edge_index


class BFG(nn.Module):
    """
    Brain-region Feature Generator
    - 每个 BFG 有独立 RGCBackbone + 读出 query q
    - 注意力读出一个脑区向量
    """
    def __init__(self, dim: int, heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.backbone = RGCBackbone(dim, heads, num_layers, dropout)
        self.q = nn.Parameter(torch.randn(dim))
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x, edge_index)        # [M,D]
        att = torch.matmul(h, self.q)           # [M]
        att = torch.softmax(att, dim=0)
        z = (att.unsqueeze(-1) * h).sum(dim=0)  # [D]
        return self.out(z)


class VBFD(nn.Module):
    """
    Visual Brain-region Feature Decomposer
    输入:
        patch_tokens: [B,M,D]  —— 上游 SGG+CLIP 提供的 patch/class token
        explicit_edges: Optional[List[edge_index]] —— SGG 显式关系
    输出:
        local_feats: [B,5,D], global_feat: [B,D]
    """
    def __init__(self, dim: int = 1024, heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1, knn_k: int = 5):
        super().__init__()
        self.dim = dim
        self.knn_k = knn_k
        self.dropout = dropout
        self.regions = nn.ModuleList([BFG(dim, heads, num_layers, dropout) for _ in REGION_LIST])
        self.global_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def merge_edges(self, x: torch.Tensor, explicit_edge_index: Optional[torch.Tensor]) -> torch.Tensor:
        knn_e = build_knn_edges(x, k=self.knn_k)
        if explicit_edge_index is None or explicit_edge_index.numel() == 0:
            return knn_e
        e = torch.cat([explicit_edge_index, knn_e], dim=1)
        uniq = torch.unique(e, dim=1)
        return uniq

    def forward(self, patch_tokens: torch.Tensor,
                explicit_edges: Optional[List[torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, M, D = patch_tokens.shape
        device = patch_tokens.device
        locals_list = []
        globals_list = []
        for b in range(B):
            x = patch_tokens[b]   # [M,D]
            e_exp = None
            if explicit_edges is not None and len(explicit_edges) > b and explicit_edges[b] is not None:
                e_exp = explicit_edges[b].to(device)
            e_all = self.merge_edges(x, e_exp)  # [2,E]

            # R 个脑区
            regs = []
            for r in range(len(REGION_LIST)):
                z_r = self.regions[r](x, e_all)  # [D]
                regs.append(z_r.unsqueeze(0))
            regs = torch.cat(regs, dim=0)        # [5,D]
            locals_list.append(regs.unsqueeze(0))

            # 全局：平均 + 投影
            g = self.global_proj(x.mean(dim=0))   # [D]
            globals_list.append(g.unsqueeze(0))

        local_feats = torch.cat(locals_list, dim=0)   # [B,5,D]
        global_feat = torch.cat(globals_list, dim=0)  # [B,D]
        return local_feats, global_feat


# -----------------------------
# 5) CLIP-style 对齐损失
# -----------------------------

class ClipLoss(nn.Module):
    def __init__(self, local_loss: bool = False):
        super().__init__()
        self.local_loss = local_loss

    def forward(self, a: torch.Tensor, b: torch.Tensor,
                logit_scale: torch.Tensor) -> torch.Tensor:
        """
        a,b: [B,D] 或 [B,5,D]（多区域时会在这里展平成 [B*5,D]）
        """
        a = l2_normalize(a, dim=-1)
        b = l2_normalize(b, dim=-1)
        if a.dim() == 3:
            B, R, D = a.shape
            a = a.reshape(B*R, D)
            b = b.reshape(B*R, D)
        logits = logit_scale.exp() * (a @ b.t())        # [N,N]
        labels = torch.arange(a.shape[0], device=a.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        return (loss_i + loss_t) * 0.5


# -----------------------------
# 6) EEG 编码主干（Transformer → WTSConv → 通道向量）
# -----------------------------

class EEGEncoder(nn.Module):
    """
    输入: eeg_raw [B,63,T]
    输出:
        node_feat: [B,63,D_node] —— 每通道一个向量
        eeg_time_feat: [B,63,T] —— WTSConv 后的时序特征（可视化等）
    """
    def __init__(self, d_model_ts: int = 128, wts_channels: int = 63,
                 wts_tk: int = 25, wave: str = "morl", pool_t: int = 5,
                 out_dim_node: int = 256):
        super().__init__()
        self.tfm = ChannelWiseTransformer(
            d_model=d_model_ts,
            nhead=8,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1
        )
        self.wts = WTSConv(channels=wts_channels, t_kernel=wts_tk,
                           wave=wave, pool_t=pool_t)
        # 对时间平均后的 [B,63,1] 做线性投影
        self.post_proj = nn.Linear(1, out_dim_node)

    def forward(self, x_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_raw: [B,63,T]
        x_t = self.tfm(x_raw)           # [B,63,T]
        x_w = self.wts(x_t)             # [B,63,T]
        x_mean = x_w.mean(dim=-1, keepdim=True)    # [B,63,1]
        node_feat = self.post_proj(x_mean).squeeze(-1)  # [B,63,D_node]
        return node_feat, x_w


# -----------------------------
# 7) 顶层模型：ATMS_CVPR26
# -----------------------------

class FGEVA(nn.Module):
    """
    整体模型：
      - EEGEncoder: raw EEG → 节点特征 (每通道向量)
      - EBFD: 超图分解 → EEG 局部(5 区) + 全局特征
      - VBFD: patch tokens → 图像局部(5 区) + 全局特征
      - ClipLoss: 全局 + 局部对齐损失
    """
    def __init__(self,
                 eeg_node_dim: int = 256,
                 eeg_local_global_dim: int = 1024,
                 img_dim: int = 1024,
                 alpha: float = 0.5):
        super().__init__()
        self.eeg_encoder = EEGEncoder(out_dim_node=eeg_node_dim)
        self.ebfd = EBFD(
            in_dim=eeg_node_dim,
            hidden_dim=256,
            out_dim=eeg_local_global_dim,
            k_pe=8,
            num_layers=2,
            dropout=0.1
        )
        self.vbfd = VBFD(
            dim=img_dim,
            heads=4,
            num_layers=2,
            dropout=0.1,
            knn_k=5
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1/0.07))
        self.alpha = alpha
        self.clip_loss = ClipLoss()

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self,
                eeg_raw: torch.Tensor,                 # [B,63,T]
                img_patch_tokens: torch.Tensor,        # [B,M,D_img]
                img_explicit_edges: Optional[List[torch.Tensor]] = None,
                electrode_coords: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        # 1) EEG 编码
        eeg_node_feat, eeg_time_feat = self.eeg_encoder(eeg_raw)           # [B,63,deeg], [B,63,T]

        # 2) EEG Brain-region Feature Decomposer
        eeg_local, eeg_global = self.ebfd(eeg_node_feat, electrode_coords) # [B,5,Dim], [B,Dim]

        # 3) Visual Brain-region Feature Decomposer
        img_local, img_global = self.vbfd(img_patch_tokens, img_explicit_edges)  # [B,5,Dim], [B,Dim]

        # 4) 对齐损失（全局 + 局部）
        Lg = self.clip_loss(eeg_global, img_global, self.logit_scale)
        Ll = self.clip_loss(eeg_local, img_local, self.logit_scale)
        loss = self.alpha * Lg + (1.0 - self.alpha) * Ll

        return {
            "loss": loss,
            "loss_global": Lg.detach(),
            "loss_local": Ll.detach(),
            "eeg_local": eeg_local,
            "eeg_global": eeg_global,
            "img_local": img_local,
            "img_global": img_global,
            "eeg_time_feat": eeg_time_feat,
        }


# -----------------------------
# 8) 简单可运行测试用例
# -----------------------------

if __name__ == "__main__":
    # 随机数据测试一遍前向，确认形状和 loss 正常
    torch.manual_seed(0)

    B, C, T = 2, 63, 250     # batch size, 通道数, 时间长度
    M, D_img = 16, 1024      # patch 数量, 图像 token 维度

    eeg_raw = torch.randn(B, C, T)
    img_patch_tokens = torch.randn(B, M, D_img)

    model = FGEVA(
        eeg_node_dim=256,
        eeg_local_global_dim=1024,
        img_dim=1024,
        alpha=0.5
    )

    out = model(eeg_raw, img_patch_tokens)

    print("loss:", out["loss"].item())
    print("loss_global:", out["loss_global"].item())
    print("loss_local:", out["loss_local"].item())
    print("eeg_local shape:", out["eeg_local"].shape)       # [B,5,1024]
    print("eeg_global shape:", out["eeg_global"].shape)     # [B,1024]
    print("img_local shape:", out["img_local"].shape)       # [B,5,1024]
    print("img_global shape:", out["img_global"].shape)     # [B,1024]
    print("eeg_time_feat shape:", out["eeg_time_feat"].shape)  # [B,63,T]
