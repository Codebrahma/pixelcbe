"""Colour-match seam-heal tiles to the finished layer-1 tiles they overlap (per-channel gain, clipped).
python heal_match.py plan_h.json heal_in_dir layer1_dir out_dir
"""
import sys, os, json
import numpy as np
from PIL import Image

ph = json.load(open(sys.argv[1])); hdir, l1dir, odir = sys.argv[2:5]; os.makedirs(odir, exist_ok=True)
p1 = json.load(open(os.environ.get('PLAN1', 'core/plan.json')))
at = {(t['tx'], t['ty']): t['name'] for t in p1['tiles']}
T = 1024
for h in ph['tiles']:
    f = f"{hdir}/{h['name']}.png"
    if not os.path.exists(f): continue
    im = np.asarray(Image.open(f).convert('RGB').resize((T, T)), np.float32)
    acc = []; hacc = []
    for dx in (-384, 384):
        for dy in (-384, 384):
            n = at.get((h['tx'] + dx, h['ty'] + dy))
            if not n or not os.path.exists(f'{l1dir}/{n}.png'): continue
            l1 = np.asarray(Image.open(f'{l1dir}/{n}.png').convert('RGB').resize((T, T)), np.float32)
            # overlap of the heal window with this tile, in each tile's own coordinates
            ox0, oy0 = max(0, -dx), max(0, -dy)          # in heal coords
            sx0, sy0 = max(0, dx), max(0, dy)            # in layer-1 coords
            w, hh = T - abs(dx), T - abs(dy)
            acc.append(l1[sy0:sy0 + hh, sx0:sx0 + w].reshape(-1, 3).mean(0))
            hacc.append(im[oy0:oy0 + hh, ox0:ox0 + w].reshape(-1, 3).mean(0))
    if acc:
        g = np.clip(np.mean(acc, 0) / np.maximum(np.mean(hacc, 0), 1), 0.85, 1.18)
        im = im * g
    Image.fromarray(im.clip(0, 255).astype(np.uint8)).save(f"{odir}/{h['name']}.png")
print('done')
