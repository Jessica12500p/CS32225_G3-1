"""Pressure appearance, marginal distributions and local orientation features."""
import numpy as np
from scipy.ndimage import gaussian_filter


def features(pressure):
    p = np.asarray(pressure, dtype=np.float32)
    if p.ndim == 2:
        p = p[None]
    if p.shape[1:] != (44,24) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('需要非负且有限的 N×44×24 压力数据')
    scale = np.maximum(p.max(axis=(1,2), keepdims=True), 1)
    q = np.sqrt(p / scale)
    smooth = gaussian_filter(q, sigma=(0,.7,.7))
    pooled = smooth.reshape(-1,22,2,12,2).mean(axis=(2,4)).reshape(len(p),-1)
    # Unsigned gradient orientation histogram in 4×4 sensor cells.
    gy, gx = np.gradient(smooth, axis=(1,2))
    magnitude = np.hypot(gx, gy)
    angles = ((np.arctan2(gy,gx) % np.pi) * 6 / np.pi).astype(int).clip(0,5)
    hist = np.stack([(magnitude*(angles==k)).reshape(-1,11,4,6,4).sum(axis=(2,4)) for k in range(6)],axis=-1)
    hist /= np.sqrt((hist**2).sum(axis=-1,keepdims=True)+.01)
    row = q.mean(axis=2); col = q.mean(axis=1)
    stats = np.stack([(p>5).mean(axis=(1,2)), np.log1p(p.mean(axis=(1,2)))/6,
                      np.log1p(p.max(axis=(1,2)))/7],axis=1)
    return np.concatenate([pooled, row, col, hist.reshape(len(p),-1)*.35, stats],axis=1).astype(np.float32)
