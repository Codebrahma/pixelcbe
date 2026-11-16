"""Forest on hills, built from the painter's own canopy: patches of dense canopy (sampled from good painted tiles)
are scattered back-to-front over forested/vegetated slopes, then hill-shaded. Roads, buildings and built-up
ground keep the painting. The model cannot paint terrain, so this replaces it on slopes.
python hill_fix.py plan.json tiles.json base_dir paint_dir out_dir
"""
import sys, os, json
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import terrain
from iso import lonlat

plan = json.load(open(sys.argv[1])); names = set(json.load(open(sys.argv[2])))
bdir, pdir, odir = sys.argv[3:6]; os.makedirs(odir, exist_ok=True)
T = 1024; S = plan['S']
HERE = os.path.dirname(os.path.abspath(__file__))
wc = np.load(f'{HERE}/../data/worldcover.npy')
WX0, WDX, WY0, WDY = 76.83, (77.08 - 76.83) / 3000, 11.13, -(11.13 - 10.97) / 1920

# --- canopy patch bank from painted dense canopy
SRC = [('core/paint6/c_24576_6144.png', None), ('core/paint6/c_26112_6144.png', None), ('core/paint6/c_25344_6144.png', None)]
bank = []
for f, _ in SRC:
    a = np.asarray(Image.open(os.path.join(HERE, f)).convert('RGB').resize((T, T)), np.float32)
    can = (a[..., 1] > a[..., 0] + 6) & (a[..., 1] > a[..., 2] + 22) & (a.mean(-1) < 125) & (a[..., 2] < 90)
    P = 72
    for y in range(0, T - P, 24):
        for x in range(0, T - P, 24):
            if can[y:y + P, x:x + P].mean() > 0.97: bank.append(a[y:y + P, x:x + P])
bank = np.array(bank)
print('canopy patches', len(bank))
yy_, xx_ = np.mgrid[0:72, 0:72]
rr_ = np.hypot(xx_ - 36, (yy_ - 36) * 1.25)
blob = np.clip((34 - rr_) / 10, 0, 1)[..., None]           # soft round crown mask

for t in plan['tiles']:
    n = t['name']
    if n not in names or not os.path.exists(f'{pdir}/{n}.png'): continue
    paint = np.asarray(Image.open(f'{pdir}/{n}.png').convert('RGB').resize((T, T)), np.float32)
    X0, Y0 = plan['X0'] + t['tx'], plan['Y0'] + t['ty']
    maxT = terrain.max_in_window(X0, Y0, T, T, S)
    if maxT >= 2:
        V, u, TT = terrain.visible(X0, Y0, T, T, S, 1, maxT)
        E = (u[None, :] + V) / 2; N = (V - u[None, :]) / 2
    else:
        from iso import ground_at
        yy, xx = np.mgrid[0:T, 0:T]
        E, N = ground_at((X0 + xx).astype(float), (Y0 + yy).astype(float), S); TT = np.zeros((T, T), np.float32)
    hill = TT > 1.0
    lo, la = lonlat(E, N)
    cls = wc[np.clip(((la - WY0) / WDY).astype(int), 0, wc.shape[0] - 1), np.clip(((lo - WX0) / WDX).astype(int), 0, wc.shape[1] - 1)]
    base = np.asarray(Image.open(f'{bdir}/{n}.png').convert('RGB'), np.float32)
    greenish = (base[..., 1] > base[..., 0] + 4) & (base[..., 1] >= base[..., 2])
    flat_tree = (~hill) & (cls == 10) & greenish
    veg = (hill & np.isin(cls, [10, 20, 30, 90, 100]) & greenish) | flat_tree
    veg = ndimage.binary_opening(veg, iterations=2)
    # lollipop trees: small isolated green blobs on open ground -> painted crowns at the same spot
    g_ = (paint[..., 1] > paint[..., 0] + 10) & (paint[..., 1] > paint[..., 2] + 10)
    lab, nl = ndimage.label(g_ & ~veg)
    lolli = []
    if nl:
        sz = ndimage.sum(g_, lab, range(1, nl + 1)); cms = ndimage.center_of_mass(g_, lab, range(1, nl + 1))
        lolli = [(int(c[0]), int(c[1]), s_) for c, s_ in zip(cms, sz) if 12 <= s_ <= 260]
    if veg.mean() < 0.01 and not lolli:
        Image.fromarray(paint.astype(np.uint8)).save(f'{odir}/{n}.png'); continue
    sh = np.ones((T, T), np.float32); sh[hill] = terrain.shade(E[hill], N[hill])
    # ground under the crowns: dark forest floor, lighter on grassland
    tex = np.zeros((T, T, 3), np.float32); tex[:] = (70, 98, 56)
    tex[cls == 30] = (176, 172, 116); tex[cls == 20] = (138, 146, 90)
    tex[flat_tree] = (150, 162, 98)
    rng = np.random.default_rng(abs(hash(n)) % 2**32)
    # crowns, back (top of screen) to front
    dens = np.where(cls == 10, 1.0, np.where(cls == 20, 0.55, 0.18)); dens = np.where(flat_tree, 0.42, dens)
    cand_y, cand_x = np.nonzero(veg[::11, ::11]); cand_y = cand_y * 11; cand_x = cand_x * 11
    order = np.argsort(cand_y + rng.random(len(cand_y)) * 10)
    for k in order:
        y, x = cand_y[k] + rng.integers(-6, 7), cand_x[k] + rng.integers(-6, 7)
        if not (0 <= y < T and 0 <= x < T) or rng.random() > dens[y, x]: continue
        pt = bank[rng.integers(len(bank))]
        y0, x0 = y - 36, x - 36
        ys0, xs0 = max(0, y0), max(0, x0); ys1, xs1 = min(T, y0 + 72), min(T, x0 + 72)
        if ys1 <= ys0 or xs1 <= xs0: continue
        pp = pt[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]; bb = blob[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
        tex[ys0:ys1, xs0:xs1] = tex[ys0:ys1, xs0:xs1] * (1 - bb) + pp * bb
    if hill.any(): tex *= (sh / max(0.5, float(sh[hill].mean())))[..., None]
    m = np.asarray(Image.fromarray((veg * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.5)), np.float32)[..., None] / 255
    out = paint * (1 - m) + tex * m
    for (y, x, s_) in sorted(lolli):
        r = int(np.clip(np.sqrt(s_) * 0.9 + 4, 6, 16))
        pt = bank[rng.integers(len(bank))]; sc = Image.fromarray(pt.astype(np.uint8)).resize((2 * r + 2, 2 * r + 2))
        yy2, xx2 = np.mgrid[0:2 * r + 2, 0:2 * r + 2]; bl = np.clip((r - np.hypot(xx2 - r - 1, (yy2 - r - 1) * 1.15)) / 2.5, 0, 1)[..., None]
        y0, x0 = y - r - 1 - 2, x - r - 1
        ys0, xs0 = max(0, y0), max(0, x0); ys1, xs1 = min(T, y0 + 2 * r + 2), min(T, x0 + 2 * r + 2)
        if ys1 <= ys0 or xs1 <= xs0: continue
        sp = np.asarray(sc, np.float32)[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]; bb = bl[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
        sy0, sy1 = min(T, ys0 + r // 2), min(T, ys1 + r // 2)            # soft ground shadow
        out[sy0:sy1, xs0 + 2:xs1 + 2 if xs1 + 2 <= T else T] *= 1 - 0.25 * bb[:sy1 - sy0, :out[sy0:sy1, xs0 + 2:min(T, xs1 + 2)].shape[1]]
        out[ys0:ys1, xs0:xs1] = out[ys0:ys1, xs0:xs1] * (1 - bb) + sp * bb
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(f'{odir}/{n}.png')
print('done')
