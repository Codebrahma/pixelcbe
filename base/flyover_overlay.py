"""Draw flyovers on top of painted tiles, procedurally (no AI), so they are always elevated and in the right place.
Input: a FLY_MASK=1 render (render_v2.py) in which every flyover part is drawn in a marker colour,
with buildings drawn normally (so whatever stands in front of a flyover still hides it).
python flyover_overlay.py tiles.json mask_dir paint_dir out_dir
"""
import sys, os, json, hashlib
import numpy as np
from PIL import Image, ImageFilter

tiles = json.load(open(sys.argv[1])); mdir, pdir, odir = sys.argv[2:5]; os.makedirs(odir, exist_ok=True)
tiles = [t['name'] if isinstance(t, dict) else t for t in (tiles['tiles'] if isinstance(tiles, dict) else tiles)]
T = 1024
MK = {'shadow': (255, 255, 0), 'girder': (0, 255, 0), 'parapet': (0, 255, 255), 'deck': (255, 0, 0),
      'pil_l': (0, 0, 255), 'pil_d': (255, 0, 255), 'dash': (255, 128, 0)}
TOL = 70


def classify(m):
    out = {}
    for k, c in MK.items():
        out[k] = np.abs(m - np.array(c, np.float32)).sum(-1) < TOL
    return out


for n in tiles:
    paint = np.asarray(Image.open(f'{pdir}/{n}.png').convert('RGB').resize((T, T)), np.float32)
    mp = f'{mdir}/{n}.png'
    if not os.path.exists(mp):
        Image.fromarray(paint.astype(np.uint8)).save(f'{odir}/{n}.png'); continue
    m = np.asarray(Image.open(mp).convert('RGB'), np.float32)
    c = classify(m)
    if sum(v.sum() for v in c.values()) < 50:
        Image.fromarray(paint.astype(np.uint8)).save(f'{odir}/{n}.png'); continue
    out = paint.copy()
    # screen-space girder: a concrete band hanging below the deck's lower edge, so even decks running up the screen
    # read as raised; pillars drop from it at intervals. Only on pixels the mask render does not cover with buildings.
    from scipy import ndimage as _nd
    deckish = c['deck'] | c['parapet'] | c['dash']
    if deckish.sum() > 200:
        GH = 9
        band = np.zeros_like(deckish)
        for k in range(1, GH + 1):
            band[k:] |= deckish[:-k]
        band &= ~deckish & ~c['girder']
        notbld = (np.abs(m - np.array([226, 214, 188], np.float32)).sum(-1) < 60) | c['shadow'] | c['pil_l'] | c['pil_d'] | (m.max(-1) - m.min(-1) < 14)
        band &= notbld
        c['girder'] = c['girder'] | band
        low = band & ~np.roll(band, -1, axis=0)
        c['shadow'] = c['shadow'] | (_nd.binary_dilation(low, structure=np.ones((5, 1)), iterations=3) & ~deckish & ~band & notbld)
        # cast shadow to the right of the deck (reads as height on decks that run up the screen)
        rs = np.zeros_like(deckish)
        for k in range(3, 17):
            rs[:, k:] |= deckish[:, :-k]
        c['shadow'] = c['shadow'] | (rs & ~deckish & ~band & notbld)
        # pillars every ~46 px along the band's lower edge
        ly, lx = np.nonzero(low)
        sel = (lx % 46) < 4
        for y, x in zip(ly[sel], lx[sel]):
            y1 = min(T, y + 16)
            c['pil_l'][y:y1, x:x + 2] = True; c['pil_d'][y:y1, x + 2:x + 4] = True
    # ground shadow: soft darkening of the painting
    sm = np.asarray(Image.fromarray((c['shadow'] * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3)), np.float32)[..., None] / 255
    out = out * (1 - 0.45 * sm)
    # pillars
    out[c['pil_l']] = (200, 194, 184); out[c['pil_d']] = (146, 140, 132)
    # girder: light concrete, darker band on its lower edge
    g = c['girder']
    out[g] = (224, 219, 208)
    below = np.zeros_like(g); below[:-3] = g[3:]
    lower_edge = g & ~np.roll(g, -3, axis=0)
    out[lower_edge] = (118, 112, 106)
    # deck asphalt with slight pixel noise
    dk = c['deck']
    yy, xx = np.nonzero(dk)
    h = ((xx // 2 * 73856093) ^ (yy // 2 * 19349663)) & 7
    out[yy, xx] = np.array([84, 85, 91], np.float32) + (h[:, None] - 3.5) * 1.5
    # parapets
    out[c['parapet']] = (238, 236, 228)
    # dashed centre line
    dy_, dx_ = np.nonzero(c['dash'])
    keep = ((dx_ + dy_) // 7) % 2 == 0
    out[dy_[keep], dx_[keep]] = (228, 224, 204); out[dy_[~keep], dx_[~keep]] = (84, 85, 91)
    core = np.asarray(Image.fromarray(((dk | c['dash']) * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(7)), np.float32) > 0
    # traffic on the deck
    rnd = np.random.default_rng(int(hashlib.md5(n.encode()).hexdigest()[:8], 16))
    cols = [(240, 196, 40), (240, 196, 40), (40, 40, 40), (220, 60, 50), (60, 110, 190), (200, 120, 40), (60, 150, 80)]
    cy, cx = np.nonzero(core)
    if len(cx):
        for k in rnd.choice(len(cx), size=min(len(cx) // 2200, 25), replace=False):
            x, y = cx[k], cy[k]
            col = cols[rnd.integers(len(cols))]
            w, hh = (5, 3) if rnd.random() < 0.75 else (9, 4)
            out[y - hh // 2:y + hh - hh // 2, x - w // 2:x + w - w // 2] = col
            out[y - hh // 2 - 1, x - w // 2:x + w - w // 2] = np.array(col) * 0.6 + 80
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(f'{odir}/{n}.png')
print('done')
