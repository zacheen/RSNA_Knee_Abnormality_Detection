"""Turn one DICOM series into a fixed-shape uint8 volume, decoding only the slices that are kept."""
import os
from pathlib import Path

import cv2
import numpy as np
import pydicom


def long_path(p) -> Path:
    """Three 64-digit UIDs push a slice path past Windows' 260-character limit. The \\\\?\\ prefix lifts it."""
    p = os.path.abspath(p)
    if os.name == "nt" and not p.startswith("\\\\?\\"):
        p = "\\\\?\\" + p
    return Path(p)


def _slice_position(ds) -> float:
    ipp = getattr(ds, "ImagePositionPatient", None)
    iop = getattr(ds, "ImageOrientationPatient", None)
    if ipp is not None and iop is not None:
        normal = np.cross(np.asarray(iop[:3], float), np.asarray(iop[3:], float))
        return float(np.dot(normal, np.asarray(ipp, float)))
    return float(getattr(ds, "InstanceNumber", 0) or 0)


def _to_float(ds) -> np.ndarray:
    img = ds.pixel_array.astype(np.float32)
    if img.ndim == 3:
        # A multi-frame or RGB object inside an MR series. Keeping the first plane is enough for a baseline.
        img = img[0] if img.shape[0] < img.shape[-1] else img[..., 0]
    img = img * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img
    return img


def _pad_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    n = max(h, w)
    out = np.full((n, n), img.min(), img.dtype)  # the slice minimum is the background level after rescaling
    out[(n - h) // 2:(n - h) // 2 + h, (n - w) // 2:(n - w) // 2 + w] = img
    return out


def _crop_square(slices: list) -> list:
    """Crop to the series' joint foreground box, then pad to square, so the resize keeps anatomy proportions.

    One box for the whole series keeps every slice aligned. Mixed slice shapes in one series only get padded.
    """
    if len({s.shape for s in slices}) != 1:
        return [_pad_square(s) for s in slices]
    proj = np.stack(slices).max(0)
    ys, xs = np.nonzero(proj > 0.1 * np.percentile(proj, 99))
    if len(ys) == 0:
        return [_pad_square(s) for s in slices]
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return [_pad_square(s[y0:y1, x0:x1]) for s in slices]


def read_series(series_dir: Path, n_slices: int, size: int, crop: bool = False) -> np.ndarray:
    """Return (n_slices, size, size) uint8, slices ordered along the acquisition normal.

    Headers are read for every file to get the order, but pixels are decoded only for the
    n_slices sampled positions. Pixel decoding dominates runtime, so this is the main speed lever.
    Without crop a 640x1280 slice is squashed to a square.
    """
    files = sorted(long_path(series_dir).glob("*.dcm"))
    if not files:
        raise FileNotFoundError(series_dir)
    positions = [_slice_position(pydicom.dcmread(f, stop_before_pixels=True)) for f in files]
    files = [files[i] for i in np.argsort(positions, kind="stable")]

    # Short series are stretched by repeating indices, so every plane yields the same depth.
    idx = np.linspace(0, len(files) - 1, n_slices).round().astype(int)
    keep = np.unique(idx)
    raw = [_to_float(pydicom.dcmread(files[i])) for i in keep]
    if crop:
        raw = _crop_square(raw)
    decoded = {i: cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA) for i, img in zip(keep, raw)}
    vol = np.stack([decoded[i] for i in idx])

    # Percentile window per volume, because intensity scales differ across scanners and sites.
    lo, hi = np.percentile(vol, [1.0, 99.5])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (vol * 255).astype(np.uint8)
