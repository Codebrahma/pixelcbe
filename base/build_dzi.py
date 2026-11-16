"""Build a Deep Zoom (DZI) pyramid straight from painted 1024 tiles, without ever holding the whole canvas.
Stitching = nearest-tile-centre ownership with a soft hem (same as stitch.py), done per 512px output tile.
python build_dzi.py plan.json paint_dir out_prefix [hem]
  -> out_prefix.dzi and out_prefix_files/<level>/<col>_<row>.webp
Missing painted tiles fall back to nothing (transparent-ish background colour).
"""
import sys, os, json, math
import numpy as np
from PIL import Image, ImageFilter

plan = json.load(open(sys.argv[1])); pdir = sys.argv[2]; outp = sys.argv[3]
HEM = int(sys.argv[4]) if len(sys.argv) > 4 else 32
HEM_E = int(os.environ.get('HEM_E', 96))   # wider blend for independently painted (FLUX) tiles
def hem_of(t): return HEM_E if t['name'].startswith(('e_', 'w', 'n_')) else HEM
W, H, T = plan['W'], plan['H'], plan['T']; tiles = plan['tiles']; TS = 512; BG = (32, 30, 28)
ext = lambda n: next((f'{pdir}/{n}{e}' for e in ('.png', '.jpg', '.webp') if os.path.exists(f'{pdir}/{n}{e}')), None)
have = [t for t in tiles if ext(t['name'])]
maxlev = math.ceil(math.log2(max(W, H)))
os.makedirs(f'{outp}_files', exist_ok=True)
cache = {}
def load(t, d=None):
    key = (d or pdir, t['name'])
    if key not in cache:
        if len(cache) > 60: cache.clear()
        f = ext(t['name']) if d is None else next((f'{d}/{t["name"]}{e}' for e in ('.png', '.jpg') if os.path.exists(f'{d}/{t["name"]}{e}')), None)
        cache[key] = np.asarray(Image.open(f).convert('RGB').resize((T, T), Image.LANCZOS), dtype=np.float32)
    return cache[key]

# optional seam-heal layer: tiles offset by half a stride, blended in along layer-1 ownership boundaries
HEAL_PLAN = os.environ.get('HEAL_PLAN'); HEAL_DIR = os.environ.get('HEAL_DIR'); BAND = float(os.environ.get('HEAL_BAND', 150))
heal = []
if HEAL_PLAN:
    heal = [t for t in json.load(open(HEAL_PLAN))['tiles'] if os.path.exists(f"{HEAL_DIR}/{t['name']}.png")]

def stitch(tl, X0, Y0, X1, Y1, d=None, hemf=None):
    near = [t for t in tl if t['tx'] < X1 and t['tx'] + T > X0 and t['ty'] < Y1 and t['ty'] + T > Y0]
    if not near: return None
    yy, xx = np.mgrid[Y0:Y1, X0:X1]
    best = np.full(xx.shape, 1e9); second = np.full(xx.shape, 1e9); owner = np.full(xx.shape, -1)
    for i, t in enumerate(near):
        dd = np.maximum(np.abs(xx - t['tx'] - T / 2), np.abs(yy - t['ty'] - T / 2))
        inside = (xx >= t['tx']) & (xx < t['tx'] + T) & (yy >= t['ty']) & (yy < t['ty'] + T)
        dd = np.where(inside, dd, 1e9)
        m = dd < best
        second = np.where(m, best, np.minimum(second, dd)); best[m] = dd[m]; owner[m] = i
    acc = np.zeros(xx.shape + (3,), np.float32); ws = np.zeros(xx.shape + (1,), np.float32)
    for i, t in enumerate(near):
        if not (owner == i).any(): continue
        hm = hemf(t) if hemf else HEM
        m = np.asarray(Image.fromarray(((owner == i) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(hm / 2)), np.float32)[..., None] / 255
        sx0, sy0 = max(X0, t['tx']), max(Y0, t['ty']); sx1, sy1 = min(X1, t['tx'] + T), min(Y1, t['ty'] + T)
        if sx1 <= sx0 or sy1 <= sy0: continue
        im = load(t, d)[sy0 - t['ty']:sy1 - t['ty'], sx0 - t['tx']:sx1 - t['tx']]
        mm = m[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0]
        acc[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0] += im * mm; ws[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0] += mm
    out = np.where(ws > 0.01, acc / np.maximum(ws, 1e-6), np.array(BG, np.float32))
    return out, ws, best, second, owner, near

def region(x0, y0, w, h):
    """stitched pixels for canvas window (x0,y0,w,h)"""
    pad = max(HEM, HEM_E) * 2 + 64
    X0, Y0, X1, Y1 = x0 - pad, y0 - pad, x0 + w + pad, y0 + h + pad
    r = stitch(have, X0, Y0, X1, Y1, None, hem_of)
    if r is None: return None
    out, ws, best, second, owner, near = r
    if heal and any(t['name'].startswith(('e_', 'w', 'n_')) for t in near):
        r2 = stitch(heal, X0, Y0, X1, Y1, HEAL_DIR, lambda t: 64)
        if r2 is not None:
            out2, ws2 = r2[0], r2[1]
            bd = np.where(second < 1e8, (second - best) / 2, 1e9)          # distance to the layer-1 seam
            is_e = np.array([t['name'].startswith(('e_', 'w', 'n_')) for t in near] + [False])
            # seam involves a FLUX tile if the owner is one, or the pixel is within the band of one
            own_e = is_e[owner]
            from scipy import ndimage
            own_e_s = ndimage.maximum_filter(own_e, size=int(BAND) * 2 + 1)
            w2 = np.clip(1 - bd / BAND, 0, 1); w2 = w2 * w2 * (3 - 2 * w2)
            w2 = w2 * own_e_s * (ws2[..., 0] > 0.5)
            out = out * (1 - w2[..., None]) + out2 * w2[..., None]
    return out[pad:pad + h, pad:pad + w]

# top level, straight from tiles
lw, lh = W, H
cols, rows = math.ceil(lw / TS), math.ceil(lh / TS)
d = f'{outp}_files/{maxlev}'; os.makedirs(d, exist_ok=True)
ONLY = None
if os.environ.get('ONLY_TILES'):   # rebuild only the top-level DZI tiles covering these plan tiles
    _on = set(json.load(open(os.environ['ONLY_TILES'])))
    ONLY = [(t['tx'], t['ty']) for t in tiles if t['name'] in _on]
def _wanted(c, r):
    if ONLY is None: return True
    x0, y0 = c * TS, r * TS
    return any(tx - TS < x0 < tx + T and ty - TS < y0 < ty + T for tx, ty in ONLY)
PART = os.environ.get('PART')            # 'k/n': build only top-level rows r % n == k, then exit
SKIPTOP = os.environ.get('SKIPTOP') == '1'
pk, pn = (map(int, PART.split('/'))) if PART else (0, 1)
for r in range(rows):
    if SKIPTOP or r % pn != pk: continue
    for c in range(cols):
        if not _wanted(c, r): continue
        w, h = min(TS, lw - c * TS), min(TS, lh - r * TS)
        px = region(c * TS, r * TS, w, h)
        im = Image.new('RGB', (w, h), BG) if px is None else Image.fromarray(px.clip(0, 255).astype(np.uint8))
        im.save(f'{d}/{c}_{r}.webp', quality=88)
    print('row', r, '/', rows, flush=True)
if PART: sys.exit(0)
# lower levels by halving
for lev in range(maxlev - 1, -1, -1):
    pw, ph = lw, lh; lw, lh = max(1, math.ceil(lw / 2)), max(1, math.ceil(lh / 2))
    pc, pr = cols, rows; cols, rows = math.ceil(lw / TS), math.ceil(lh / TS)
    d = f'{outp}_files/{lev}'; os.makedirs(d, exist_ok=True)
    sc = 2 ** (maxlev - lev)
    for r in range(rows):
        for c in range(cols):
            if ONLY is not None and not any(tx - TS * sc < c * TS * sc < tx + T and ty - TS * sc < r * TS * sc < ty + T for tx, ty in ONLY): continue
            w, h = min(TS, lw - c * TS), min(TS, lh - r * TS)
            big = Image.new('RGB', (w * 2, h * 2), BG)
            for dy in (0, 1):
                for dx in (0, 1):
                    p = f'{outp}_files/{lev + 1}/{2 * c + dx}_{2 * r + dy}.webp'
                    if os.path.exists(p): big.paste(Image.open(p), (dx * TS, dy * TS))
            big.resize((w, h), Image.LANCZOS).save(f'{d}/{c}_{r}.webp', quality=88)
open(f'{outp}.dzi', 'w').write(f'<?xml version="1.0" encoding="UTF-8"?><Image xmlns="http://schemas.microsoft.com/deepzoom/2008" Format="webp" Overlap="0" TileSize="{TS}"><Size Width="{W}" Height="{H}"/></Image>')
print('done', outp)
