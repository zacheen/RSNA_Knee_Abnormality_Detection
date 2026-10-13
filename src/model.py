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
    """Read a checkpoint, defaulting the keys older ones lack (they were uncropped, 3-slot and used attn pooling)."""
    return {"crop_fg": False, "pool": "attn", "planes": ALL_SLOTS[:3], **torch.load(path, map_location="cpu")}


def build_from_checkpoint(ck: dict) -> "KneeNet":
    """KneeNet with the checkpoint's architecture and weights, without a pretrained download (scoring is offline)."""
    model = KneeNet(ck["backbone"], pretrained=False, pool=ck["pool"], n_slots=len(ck["planes"]))
    model.load_state_dict(ck["state_dict"])
    return model


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
