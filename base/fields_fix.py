"""Draw open fields procedurally where the base render shows field ground (WorldCover crop/grass/shrub or OSM
farmland/meadow), so the painter cannot invent towns on open land. Roads, buildings and trees keep the painting.
python fields_fix.py plan.json tiles.json base_dir paint_dir out_dir
"""
import sys, os, json
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iso import ground_at, lonlat

plan = json.load(open(sys.argv[1])); names = set(json.load(open(sys.argv[2])))
bdir, pdir, odir = sys.argv[3:6]; os.makedirs(odir, exist_ok=True)
T = 1024; S = plan['S']
wc = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/worldcover.npy'))
WX0, WDX, WY0, WDY = 76.83, (77.08 - 76.83) / 3000, 11.13, -(11.13 - 10.97) / 1920
FIELD = np.array([196, 196, 140], np.float32)
n_done = 0
for t in plan['tiles']:
    n = t['name']
    if n not in names or not os.path.exists(f'{pdir}/{n}.png'): continue
    base = np.asarray(Image.open(f'{bdir}/{n}.png').convert('RGB'), np.float32)
    paint = np.asarray(Image.open(f'{pdir}/{n}.png').convert('RGB').resize((T, T)), np.float32)
    fm = np.abs(base - FIELD).sum(-1) < 14
    edu = np.abs(base - np.array([218, 206, 174], np.float32)).sum(-1) < 14
    fm = ndimage.binary_opening(fm, iterations=3)
    lab, nl = ndimage.label(fm)
    if nl:
        sz = ndimage.sum(fm, lab, range(1, nl + 1)); fm = np.isin(lab, 1 + np.nonzero(sz > 1500)[0])
    yy, xx = np.mgrid[0:T, 0:T]
    gx = xx + plan['X0'] + t['tx']; gy = yy + plan['Y0'] + t['ty']
    e, nn = ground_at(gx.astype(float), gy.astype(float), S)
    lo, la = lonlat(e, nn)
    c = np.clip(((lo - WX0) / WDX).astype(int), 0, wc.shape[1] - 1); r = np.clip(((la - WY0) / WDY).astype(int), 0, wc.shape[0] - 1)
    cls = wc[r, c]
    eduveg = ndimage.binary_opening(edu & np.isin(cls, [10, 20, 30]), iterations=3)
    fm = eduveg if os.environ.get('EDU_ONLY') == '1' else (fm | eduveg)
    cls = np.where(eduveg & (cls == 10), 20, cls)
    if fm.mean() < 0.01:
        Image.fromarray(paint.astype(np.uint8)).save(f'{odir}/{n}.png'); continue
    # field parcels ~ 40 m x 60 m in ground coordinates, each with its own crop/tone and row direction
    pe, pn = np.floor(e / 40).astype(np.int64), np.floor(nn / 55).astype(np.int64)
    ph = ((pe * 73856093) ^ (pn * 19349663)) & 0xFFFF
    tones = np.array([(176, 184, 112), (160, 176, 100), (190, 186, 124), (200, 182, 130), (150, 170, 96), (186, 170, 120)], np.float32)
    tex = tones[ph % len(tones)].copy()
    rows_ = ((np.floor(e * (1 + (ph % 2)) + nn * (ph % 2 == 0)) / 2.2).astype(np.int64) % 2 == 0) & (cls == 40)
    tex[rows_] *= 0.92
    # dry grass / scrub: lighter, speckled
    grass = cls != 40
    hh = (((gx // 2) * 2654435761) ^ ((gy // 2) * 40503)) & 15
    lf = (np.sin(e * 0.031 + nn * 0.017) + np.sin(e * 0.011 - nn * 0.027) + np.sin((e + 2 * nn) * 0.047)) / 3
    gcol = np.array([178, 178, 120], np.float32) * (1 - (lf[..., None] + 1) / 2 * 0.0) + 0
    gcol = gcol + lf[..., None] * np.array([-14, 4, -10], np.float32)
    tex[grass] = gcol[grass] + ((hh[grass] - 7.5) * 0.9)[:, None]
    # parcel bunds (crop only)
    be = (np.abs(e - np.round(e / 40) * 40) < 0.7) | (np.abs(nn - np.round(nn / 55) * 55) < 0.7)
    tex[be & (cls == 40)] = (150, 136, 98)
    # bush sprites on a jittered 7 m ground grid
    from PIL import ImageDraw
    timg = Image.fromarray(tex.clip(0, 255).astype(np.uint8)); dd = ImageDraw.Draw(timg, 'RGBA')
    rng = np.random.default_rng(abs(hash(n)) % 2**32)
    ys, xs = np.nonzero(fm[::6, ::6]); ys = ys * 6; xs = xs * 6
    for k in rng.choice(len(xs), size=min(len(xs) // 4, 1600), replace=False) if len(xs) else []:
        x, y = int(xs[k] + rng.integers(-3, 4)), int(ys[k] + rng.integers(-3, 4)); r = int(rng.integers(3, 7))
        dd.ellipse([x - r + 2, y - r // 2 + 3, x + r + 2, y + r // 2 + 4], fill=(0, 0, 0, 50))
        dd.ellipse([x - r, y - r, x + r, y + r * 0.8], fill=(96, 128, 70, 255))
        dd.ellipse([x - r * 0.6, y - r * 0.9, x + r * 0.3, y], fill=(124, 156, 86, 255))
    tex = np.asarray(timg, np.float32)
    # keep painted tree canopy (the model's trees are better than anything procedural)
    canopy = (paint[..., 1] > paint[..., 0] + 8) & (paint[..., 1] > paint[..., 2] + 8) & (paint.mean(-1) < 130)
    canopy = ndimage.binary_opening(canopy, iterations=1)
    fm = fm & ~ndimage.binary_dilation(canopy, iterations=1)
    roof = (paint[..., 0] > paint[..., 1] + 35) | ((paint.max(-1) - paint.min(-1) < 16) & (paint.mean(-1) > 95) & (paint.mean(-1) < 215))
    roof = ndimage.binary_opening(roof, iterations=2)
    lab2, nl2 = ndimage.label(roof)
    if nl2:
        sz2 = ndimage.sum(roof, lab2, range(1, nl2 + 1)); roof = np.isin(lab2, 1 + np.nonzero(sz2 > 60)[0])
    fm = fm & ~ndimage.binary_dilation(roof, iterations=2)
    m = np.asarray(Image.fromarray((fm * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)), np.float32)[..., None] / 255
    out = paint * (1 - m) + tex * m
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(f'{odir}/{n}.png'); n_done += 1
print('field tiles', n_done)
