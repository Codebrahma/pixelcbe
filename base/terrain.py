"""Terrain support for the iso renderer: Copernicus GLO-30 heights above the city plain, floating-horizon
visibility per screen pixel, and hill shading. Heights are relative to BASE (the plain stays flat, h=0).
"""
import os, math
import numpy as np
from iso import lonlat, C30

BASE = 452.0          # metres; city plain ~430-450 stays flat
_D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
_dem = np.load(f'{_D}/dem_n11.npy').astype(np.float32)
_tr = list(map(float, open(f'{_D}/dem_n11_transform.txt').read().split()))   # a, b, c, d, e, f (affine)
_a, _c, _e, _f = _tr[0], _tr[2], _tr[4], _tr[5]
# light smoothing so 30 m cells do not show as steps
from scipy import ndimage as _nd
_dem = _nd.gaussian_filter(_dem, 0.8)


def ter(e, n):
    """height above the plain (m) at ground coords (vectorised), bilinear, 0 outside the DEM / on the plain"""
    lo, la = lonlat(np.asarray(e, np.float64), np.asarray(n, np.float64))
    cf = (lo - _c) / _a - 0.5; rf = (la - _f) / _e - 0.5
    c0 = np.floor(cf).astype(int); r0 = np.floor(rf).astype(int)
    ok = (c0 >= 0) & (c0 < _dem.shape[1] - 1) & (r0 >= 0) & (r0 < _dem.shape[0] - 1)
    c0c = np.clip(c0, 0, _dem.shape[1] - 2); r0c = np.clip(r0, 0, _dem.shape[0] - 2)
    fx = cf - c0c; fy = rf - r0c
    h = (_dem[r0c, c0c] * (1 - fx) * (1 - fy) + _dem[r0c, c0c + 1] * fx * (1 - fy) +
         _dem[r0c + 1, c0c] * (1 - fx) * fy + _dem[r0c + 1, c0c + 1] * fx * fy)
    h = np.where(ok, h - BASE, 0.0)
    # soft knee so the plain/foothill transition is smooth
    return np.where(h > 0, h * np.clip(h / 25.0, 0, 1), 0.0)


def max_in_window(X0, Y0, W, H, S, extra_px=2400):
    """largest terrain height that could rise into this screen window (samples ground below it)"""
    xs = np.linspace(X0, X0 + W, 24); ys = np.linspace(Y0, Y0 + H + extra_px, 60)
    gx, gy = np.meshgrid(xs, ys)
    a = gx / (C30 * S); b = -gy / (0.5 * S)
    e = (a + b) / 2; n = (b - a) / 2
    return float(ter(e, n).max())


def visible(X0, Y0, W, H, S, SS, maxT, dv=0.5):
    """floating horizon at supersampled resolution: returns v (=e+n of the visible ground point), u per column,
    and terrain height T at each visible point. Arrays are (H*SS, W*SS)."""
    Wp, Hp = W * SS, H * SS
    xs = X0 + (np.arange(Wp) + 0.5) / SS
    u = xs / (C30 * S)                                   # e - n, constant per column
    v_lo = -2 * (Y0 + H) / S - 2 * maxT - 4             # nearest ground that can reach the window bottom
    v_hi = -2 * Y0 / S + 4                               # farthest (flat ground at window top)
    vs = np.arange(v_lo, v_hi, dv)
    V = np.empty((Hp, Wp), np.float32); TT = np.empty((Hp, Wp), np.float32)
    rows = np.arange(Hp)
    for c0 in range(0, Wp, 128):
        uu = u[c0:c0 + 128][None, :]
        vv = vs[:, None]
        e = (uu + vv) / 2; n = (vv - uu) / 2
        T = ter(e, n)                                    # (len(vs), cols)
        ys = (-(0.5 * vv + T) * S - Y0) * SS             # screen row (supersampled) of each sample
        cm = np.minimum.accumulate(ys, axis=0)           # horizon after each step (near -> far)
        for j in range(cm.shape[1]):
            col = cm[:, j]
            # first index where horizon <= row  ->  searchsorted on the negated (non-decreasing) sequence
            idx = np.searchsorted(-col, -rows - 0.5, side='left')
            idx = np.clip(idx, 0, len(vs) - 1)
            V[:, c0 + j] = vs[idx]; TT[:, c0 + j] = T[idx, j]
    return V, u, TT


def shade(e, n, eps=6.0):
    """hillshade factor (~0.7..1.15), light from the west-south-west, high sun"""
    hx = (ter(e + eps, n) - ter(e - eps, n)) / (2 * eps)
    hy = (ter(e, n + eps) - ter(e, n - eps)) / (2 * eps)
    nx, ny, nz = -hx, -hy, 1.0
    L = np.array([-0.55, -0.35, 0.76]); L = L / np.linalg.norm(L)
    d = (nx * L[0] + ny * L[1] + nz * L[2]) / np.sqrt(nx * nx + ny * ny + nz * nz)
    return np.clip(0.25 + 1.0 * d, 0.45, 1.3)
