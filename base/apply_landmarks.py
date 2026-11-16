"""Blend accepted landmark paintings into the DZI pyramid (level-16 tiles), then rebuild the parent levels for the
touched area. Only the region that changed around the centre is taken from the painting.
python apply_landmarks.py accepted.json dzi_files_dir      (accepted.json: [{"id":..., "out": path}, ...])
"""
import sys, os, json, math
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

acc = json.load(open(sys.argv[1])); D = sys.argv[2].rstrip('/'); TS = 512
maxlev = max(int(x) for x in os.listdir(D) if x.isdigit())
touched = set()
for a in acc:
    pos = json.load(open(f"/tmp/lm/{a['id']}/pos.json")); x0, y0 = pos['x0'], pos['y0']
    crop = np.asarray(Image.open(f"/tmp/lm/{a['id']}/crop.png").convert('RGB'), np.float32)
    out = np.asarray(Image.open(a['out']).convert('RGB').resize((1024, 1024), Image.LANCZOS), np.float32)
    # colour-match the painting to the crop on the outer ring (Gemini drifts colours slightly)
    yy, xx = np.mgrid[0:1024, 0:1024]; rr = np.hypot(xx - 512, (yy - 560) * 1.3)
    ring = rr > 430
    g = np.clip(crop[ring].mean(0) / np.maximum(out[ring].mean(0), 1), 0.85, 1.15); out = out * g
    diff = np.abs(out - crop).sum(-1)
    ch = ndimage.binary_opening(ndimage.median_filter((diff > 60).astype(np.uint8), 5) > 0, iterations=2)
    ch &= rr < float(a.get('radius', 380))
    lab, n = ndimage.label(ndimage.binary_closing(ch, iterations=6))
    if n:   # keep components that reach near the centre
        keep = [i + 1 for i in range(n) if (np.hypot(*(np.array(ndimage.center_of_mass(lab == i + 1)) - [560, 512])) < 300 and (lab == i + 1).sum() > 400)]
        ch = np.isin(lab, keep)
    m = ndimage.binary_fill_holes(ndimage.binary_dilation(ch, iterations=10))
    m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6)), np.float32)[..., None] / 255
    res = crop * (1 - m) + out * m
    print(a['id'], 'mask px', int((m > 0.5).sum()))
    for r in range(y0 // TS, (y0 + 1024) // TS + 1):
        for c in range(x0 // TS, (x0 + 1024) // TS + 1):
            f = f'{D}/{maxlev}/{c}_{r}.webp'
            if not os.path.exists(f): continue
            t = np.asarray(Image.open(f).convert('RGB'), np.float32).copy()
            ox, oy = c * TS - x0, r * TS - y0
            sx0, sy0 = max(0, ox), max(0, oy); sx1, sy1 = min(1024, ox + t.shape[1]), min(1024, oy + t.shape[0])
            if sx1 <= sx0 or sy1 <= sy0: continue
            mm = m[sy0:sy1, sx0:sx1]
            if mm.max() < 0.01: continue
            t[sy0 - oy:sy1 - oy, sx0 - ox:sx1 - ox] = res[sy0:sy1, sx0:sx1] * mm + t[sy0 - oy:sy1 - oy, sx0 - ox:sx1 - ox] * (1 - mm)
            Image.fromarray(t.clip(0, 255).astype(np.uint8)).save(f, quality=88); touched.add((c, r))
# rebuild parents of touched tiles
cur = touched
for lev in range(maxlev - 1, -1, -1):
    par = {(c // 2, r // 2) for c, r in cur}
    for c, r in par:
        kids = [(2 * c + dx, 2 * r + dy) for dy in (0, 1) for dx in (0, 1)]
        ims = {k: Image.open(f'{D}/{lev + 1}/{k[0]}_{k[1]}.webp') for k in kids if os.path.exists(f'{D}/{lev + 1}/{k[0]}_{k[1]}.webp')}
        if not ims: continue
        old = Image.open(f'{D}/{lev}/{c}_{r}.webp'); w, h = old.size
        big = Image.new('RGB', (w * 2, h * 2), (32, 30, 28))
        for (kc, kr), im in ims.items(): big.paste(im, ((kc - 2 * c) * TS, (kr - 2 * r) * TS))
        big.resize((w, h), Image.LANCZOS).save(f'{D}/{lev}/{c}_{r}.webp', quality=88)
    cur = par
print('touched level-max tiles', len(touched))
