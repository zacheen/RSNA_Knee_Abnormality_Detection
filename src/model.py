"""2.5D study classifier. A shared 2D backbone embeds every slice, attention pools each slot (one series) over depth,
and the slot embeddings are concatenated into 12 logits."""
import numpy as np
import timm
import torch
import torch.nn as nn

from config import ALL_SLOTS, LABELS, SLOT_SPECS


def to_input(vol: np.ndarray) -> torch.Tensor:
    """uint8 cache volume to the float [0, 1] tensor KneeNet expects. Training and inference both go through here."""
    return torch.from_numpy(vol).float().div_(255.0)


def mirror_lr(vol: torch.Tensor) -> torch.Tensor:
    """Mirror patient left-right on a [..., P, D, H, W] volume, whose P slots are ALL_SLOTS[:P]. No label is swapped.

    A mirrored left knee is a right knee and its medial meniscus is still medial.
    Coronal and axial rows run left-right, so they flip in width. Sagittal slices are stacked left-right, so they
    flip in depth. Flipping sagittal width would mirror front to back, which no real knee looks like.
    AttnPool ignores slice order, so the sagittal depth flip is a no-op under pool="attn" but not under "conv".
    """
    assert vol.shape[-4] <= len(ALL_SLOTS), vol.shape
    out = vol.clone()
    for i, slot in enumerate(ALL_SLOTS[: vol.shape[-4]]):
        out[..., i, :, :, :] = vol[..., i, :, :, :].flip(-3 if SLOT_SPECS[slot][0] == "Sagittal" else -1)
    return out


def predict_tta(model: nn.Module, vol: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    """Sigmoid probabilities averaged over the volume and its left-right mirror.

    Logits are cast to fp32 first. Under autocast an fp16 sigmoid collapses distinct logits into ties, and AUC
    only sees ranks.
    """
    return (torch.sigmoid(model(vol, present).float()) + torch.sigmoid(model(mirror_lr(vol), present).float())) / 2


class AttnPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, 1))

    def forward(self, x):  # x [B, D, F]
        w = torch.softmax(self.score(x), dim=1)
        return (w * x).sum(1)


class DepthConvPool(nn.Module):
    """Residual 1D convs over the ordered slices, then AttnPool, which alone ignores slice order.

    The last conv is zero-initialized, so this starts exactly as AttnPool.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(dim, dim, 3, padding=1), nn.GELU(), nn.Conv1d(dim, dim, 3, padding=1))
        nn.init.zeros_(self.conv[2].weight)
        nn.init.zeros_(self.conv[2].bias)
        self.attn = AttnPool(dim)

    def forward(self, x):  # x [B, D, F]
        return self.attn(x + self.conv(x.transpose(1, 2)).transpose(1, 2))


POOLS = {"attn": AttnPool, "conv": DepthConvPool}


def load_checkpoint(path) -> dict:
    """Read a checkpoint, defaulting the keys older ones lack (they were uncropped, 3-slot, used attn pooling, spread
    slices over the whole stack and were the per-slice KneeNet)."""
    return {"crop_fg": False, "pool": "attn", "planes": ALL_SLOTS[:3], "crop_mm": 0.0, "band": [0.0, 1.0],
            "arch": "slices", **torch.load(path, map_location="cpu")}


def build_from_checkpoint(ck: dict) -> nn.Module:
    """The checkpoint's architecture with its weights, without a pretrained download (scoring is offline)."""
    if ck["arch"] == "windows":
        model = WindowNet(ck["backbone"], pretrained=False, n_slots=len(ck["planes"]), **ck["window_args"])
    else:
        model = KneeNet(ck["backbone"], pretrained=False, pool=ck["pool"], n_slots=len(ck["planes"]))
    model.load_state_dict(ck["state_dict"])
    return model


class WindowNet(nn.Module):
    """Three adjacent cached slices form one RGB window, every window is encoded, and each finding attends over all
    windows of all slots with its own query.

    Windows never cross a slot boundary. In training the crop position is random and only train_windows windows
    are encoded per step, which keeps memory bounded. In eval every window is used with a centre crop.
    """

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, backbone: str = "convnext_tiny", pretrained: bool = True, n_slots: int = 5, crop: int = 288,
                 train_windows: int = 32, dim: int = 384, chunk: int = 64):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained, in_chans=3, num_classes=0)
        f = self.encoder.num_features
        self.n_slots, self.crop, self.train_windows, self.chunk = n_slots, crop, train_windows, chunk
        self.proj = nn.Sequential(nn.Linear(f, dim), nn.LayerNorm(dim))
        self.slot_emb = nn.Embedding(n_slots, dim)
        self.pos_emb = nn.Linear(1, dim)
        nn.init.zeros_(self.slot_emb.weight)
        nn.init.zeros_(self.pos_emb.weight)
        nn.init.zeros_(self.pos_emb.bias)
        self.score = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, len(LABELS)))
        self.cls_w = nn.Parameter(torch.randn(len(LABELS), dim) * dim ** -0.5)
        self.cls_b = nn.Parameter(torch.zeros(len(LABELS)))
        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, vol, present):
        # vol [B, P, D, H, W] float in [0, 1]; present [B, P] bool
        vol, present = vol[:, : self.n_slots], present[:, : self.n_slots]
        b, p, d, h, w = vol.shape
        c = min(self.crop, h, w)
        if self.training:
            y0, x0 = np.random.randint(0, h - c + 1), np.random.randint(0, w - c + 1)
        else:
            y0, x0 = (h - c) // 2, (w - c) // 2
        vol = vol[..., y0:y0 + c, x0:x0 + c]
        slot = torch.arange(p, device=vol.device).repeat_interleave(d - 2)
        centre = torch.arange(1, d - 1, device=vol.device).repeat(p)
        if self.training and self.train_windows < len(slot):
            keep = torch.randperm(len(slot), device=vol.device)[: self.train_windows].sort().values
            slot, centre = slot[keep], centre[keep]
        idx = centre[:, None] + torch.arange(-1, 2, device=vol.device)  # [W, 3]
        win = vol[:, slot[:, None], idx]  # [B, W, 3, c, c]
        x = (win.reshape(-1, 3, c, c) - self.mean) / self.std
        feats = torch.cat([self.encoder(x[i:i + self.chunk]) for i in range(0, len(x), self.chunk)])
        tok = self.proj(feats.reshape(b, len(slot), -1)) + self.slot_emb(slot) + self.pos_emb((centre / (d - 1))[:, None].to(feats.dtype))
        valid = present[:, slot]  # [B, W]
        a = self.score(tok).float().masked_fill(~valid[..., None], float("-inf"))
        a = torch.nan_to_num(torch.softmax(a, dim=1))  # a study with no slot at all gets zero weights, not NaN
        pooled = torch.einsum("bwk,bwd->bkd", a, tok.float())
        return (pooled * self.cls_w).sum(-1) + self.cls_b


class KneeNet(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, dropout: float = 0.25, pool: str = "attn",
                 n_slots: int = 3):
        super().__init__()
        # in_chans=1 makes timm sum the RGB stem weights, so ImageNet pretraining still transfers to grayscale.
        self.encoder = timm.create_model(backbone, pretrained=pretrained, in_chans=1, num_classes=0)
        f = self.encoder.num_features
        self.n_slots = n_slots
        self.pools = nn.ModuleList(POOLS[pool](f) for _ in range(n_slots))
        self.drop = nn.Dropout(dropout)  # no parameters, so checkpoints saved without it still load
        self.head = nn.Linear(f * n_slots, len(LABELS))

    def forward(self, vol, present):
        # vol [B, P, D, H, W] float in [0, 1]; present [B, P] bool. Slots past n_slots are dropped, so a 3-slot
        # checkpoint reads a 5-slot volume. This relies on the slot-prefix invariant in config.py.
        vol, present = vol[:, : self.n_slots], present[:, : self.n_slots]
        b, p, d, h, w = vol.shape
        feats = self.encoder(vol.reshape(b * p * d, 1, h, w)).reshape(b, p, d, -1)
        planes = torch.stack([self.pools[i](feats[:, i]) for i in range(p)], 1)
        planes = planes * present.unsqueeze(-1).to(planes.dtype)  # a missing plane contributes zeros, not noise
        return self.head(self.drop(planes.flatten(1)))
