"""Stitch painted tiles: each pixel belongs to the tile whose centre is nearest (L-inf),
with a short soft 'hem' along the cut lines (Justin Chavez's hemming idea).
python stitch.py tiles.json paint_dir W H out.png [hem_px]
"""
import sys, json
import numpy as np
from PIL import Image, ImageFilter

tiles = json.load(open(sys.argv[1])); pdir = sys.argv[2]
W, H = int(sys.argv[3]), int(sys.argv[4]); out = sys.argv[5]
HEM = int(sys.argv[6]) if len(sys.argv) > 6 else 32
T = 1024
yy, xx = np.mgrid[0:H, 0:W]
best = np.full((H, W), 1e9); owner = np.full((H, W), -1)
for i, t in enumerate(tiles):
    cx, cy = t['tx'] + T / 2, t['ty'] + T / 2
    d = np.maximum(np.abs(xx - cx), np.abs(yy - cy))
    inside = (xx >= t['tx']) & (xx < t['tx'] + T) & (yy >= t['ty']) & (yy < t['ty'] + T)
    d = np.where(inside, d, 1e9)
    m = d < best; best[m] = d[m]; owner[m] = i
acc = np.zeros((H, W, 3)); wsum = np.zeros((H, W, 1))
for i, t in enumerate(tiles):
    mask = Image.fromarray(((owner == i) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(HEM / 2))
    m = np.asarray(mask, dtype=float)[..., None] / 255
    x, y = t['tx'], t['ty']
    sub = np.zeros((H, W, 1)); sub[y:y + T, x:x + T] = m[y:y + T, x:x + T]
    im = np.asarray(Image.open(f"{pdir}/{t['name']}.png").convert('RGB').resize((T, T), Image.LANCZOS), dtype=float)
    acc[y:y + T, x:x + T] += im * sub[y:y + T, x:x + T]; wsum += sub
Image.fromarray((acc / np.maximum(wsum, 1e-6)).clip(0, 255).astype(np.uint8)).save(out)
print('saved', out)
