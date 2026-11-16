"""Replace AI-painted water with one consistent pixel-art water texture, using OSM water from the base render as mask.
Roads/rails/bridges/buildings over water stay (the base render shows them as non-water there).
python water_fix.py plan.json base_dir paint_dir out_dir
"""
import sys, os, os, json, hashlib
import numpy as np
from PIL import Image, ImageFilter

plan = json.load(open(sys.argv[1])); bdir, pdir, odir = sys.argv[2:5]; os.makedirs(odir, exist_ok=True)
T = plan['T']
WATER_BASE = np.array([86, 146, 186], float)
DEEP = np.array([58, 118, 168], float); MID = np.array([70, 134, 182], float)
HI = np.array([150, 198, 226], float); EDGE = np.array([48, 100, 146], float)

def texture(gx, gy):
    """deterministic global water texture for canvas coords gx, gy (2D int arrays)"""
    # gentle large-scale variation, in 8px pixel-art blocks
    bx, by = gx // 8, gy // 8
    h = ((bx * 73856093) ^ (by * 19349663)) & 0xFFFF
    v = (np.sin(bx * 0.11 + by * 0.07) + np.sin(bx * 0.05 - by * 0.13)) * 0.12 + 0.5
    col = DEEP[None, None] * (1 - v[..., None]) + MID[None, None] * v[..., None]
    col += ((h % 7) - 3)[..., None] * 1.2
    # ripples: short horizontal light dashes on a sparse grid
    cx, cy = gx // 24, gy // 10
    hh = ((cx * 2654435761) ^ (cy * 40503)) & 0xFFFFFFFF
    on = (hh % 4 == 0)
    off = (hh >> 8) % 14
    lx = (gx % 24); ly = (gy % 10)
    dash = on & (ly == 4) & (lx >= off % 10) & (lx < off % 10 + 6 + (hh >> 4) % 6)
    col[dash] = HI
    return col

for t in plan['tiles']:
    name = t['name']; out = f'{odir}/{name}.png'
    base = np.asarray(Image.open(f'{bdir}/{name}.png').convert('RGB'), float)
    paint = np.asarray(Image.open(f'{pdir}/{name}.png').convert('RGB').resize((T, T)), float)
    wm = (np.abs(base - WATER_BASE).sum(-1) < 30)
    if wm.mean() < 0.002:
        Image.fromarray(paint.astype(np.uint8)).save(out); continue
    from scipy import ndimage
    wm = ndimage.binary_fill_holes(wm)  # objects fully surrounded by water (stray footprints) become water; bridges touch shore so stay
    m = Image.fromarray((wm * 255).astype(np.uint8))
    m = m.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))  # drop specks
    hard = np.asarray(m, float) / 255
    inner = np.asarray(m.filter(ImageFilter.MinFilter(5)), float) / 255
    soft = np.asarray(m.filter(ImageFilter.GaussianBlur(2.5)), float)[..., None] / 255
    gy, gx = np.mgrid[0:T, 0:T]; gx = gx + plan['X0'] + t['tx']; gy = gy + plan['Y0'] + t['ty']
    tex = texture(gx, gy)
    # shallow, lighter water near the shore
    shallow = 1 - np.asarray(m.filter(ImageFilter.MinFilter(9)).filter(ImageFilter.GaussianBlur(10)), float) / 255
    SH = np.array([112, 172, 204], float)
    tex = tex * (1 - 0.7 * shallow[..., None]) + SH * 0.7 * shallow[..., None]
    edge = (hard > 0.5) & (inner < 0.5)
    tex[edge] = tex[edge] * 0.4 + EDGE * 0.6
    # floating vegetation / marsh where the aligned aerial photo shows green on the water
    GD = os.environ.get('VEG_DIR')   # optional: per-tile images where green-over-water marks lake vegetation
    if GD and os.path.exists(f'{GD}/{name}.jpg'):
        from scipy import ndimage
        g = np.asarray(Image.open(f'{GD}/{name}.jpg').convert('RGB').resize((T, T)), float)
        veg = (g[..., 1] > g[..., 0] + 18) & (g[..., 1] > g[..., 2] + 6) & (hard > 0.5)
        veg = ndimage.median_filter(veg.astype(np.uint8), size=9) > 0
        veg = ndimage.binary_opening(veg, iterations=3)
        lab, nl = ndimage.label(veg)
        if nl:
            sizes = ndimage.sum(veg, lab, range(1, nl + 1)); veg = np.isin(lab, 1 + np.nonzero(sizes > 400)[0])
        if veg.any():
            hv = (((gx // 3) * 2654435761) ^ ((gy // 3) * 40503)) & 15
            vt = np.zeros_like(tex); vt[:] = (84, 124, 66)
            vt[hv < 5] = (104, 146, 76); vt[hv == 7] = (132, 162, 90); vt[(hv == 11) & (((gx + gy) // 2) % 3 == 0)] = (196, 170, 210)   # hyacinth flowers
            vs = np.asarray(Image.fromarray((veg * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5)), float)[..., None] / 255
            rim = veg & ~ndimage.binary_erosion(veg, iterations=2)
            vt[rim] = (62, 96, 54)
            tex = tex * (1 - vs) + vt * vs
    res = paint * (1 - soft) + tex * soft
    # railway embankments across / beside water: draw procedurally (the painter tends to turn them into roads)
    from scipy import ndimage as _nd
    ball = (np.abs(base - np.array([130, 118, 104])).sum(-1) < 22)
    railc = (np.abs(base - np.array([70, 66, 62])).sum(-1) < 22) & _nd.binary_dilation(ball, iterations=2)
    near_w = _nd.binary_dilation(hard > 0.5, iterations=30)
    rail = (ball | railc) & near_w
    if rail.sum() > 200:
        bund = _nd.binary_dilation(rail, iterations=5) & ~rail
        hh = (((gx // 2) * 73856093) ^ ((gy // 2) * 19349663)) & 7
        res[bund] = np.array([160, 140, 106]) + ((hh[bund] - 3.5) * 2)[:, None]
        edge_b = bund & ~_nd.binary_erosion(_nd.binary_dilation(rail, iterations=5), iterations=1)
        res[edge_b] = (104, 132, 80)
        res[ball & near_w] = np.array([132, 124, 112]) + ((hh[ball & near_w] - 3.5) * 2.5)[:, None]
        rc = railc & near_w
        sleeper = rc & (((gx + gy) // 3) % 2 == 0)
        res[rc] = (64, 60, 58); res[sleeper] = (112, 92, 70)
    Image.fromarray(res.clip(0, 255).astype(np.uint8)).save(out)
print('done')
