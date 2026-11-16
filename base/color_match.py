"""Harmonise tile colours: per-tile per-channel gain so overlap bands agree with neighbours (iterative).
python color_match.py plan.json in_dir out_dir [strength]"""
import sys, os, json
import numpy as np
from PIL import Image
plan = json.load(open(sys.argv[1])); idir, odir = sys.argv[2:4]; os.makedirs(odir, exist_ok=True)
STR = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8
FIXED = set(json.load(open(os.environ['FIXED']))) if os.environ.get('FIXED') else set()
T = 1024; tiles = plan['tiles']; at = {(t['tx'], t['ty']): t for t in tiles}
small = {t['name']: np.asarray(Image.open(f"{idir}/{t['name']}.png").convert('RGB').resize((256, 256)), float) for t in tiles}
S = 4  # downscale factor
pairs = []
for t in tiles:
    for dx, dy in ((768, 0), (0, 768), (768, 768), (768, -768)):
        n = at.get((t['tx'] + dx, t['ty'] + dy))
        if not n: continue
        ox, oy = dx // S, dy // S; w = T // S
        ax0, ay0 = max(0, ox), max(0, oy); ax1, ay1 = min(w, ox + w), min(w, oy + w)
        a = small[t['name']][ay0:ay1, ax0:ax1].reshape(-1, 3).mean(0)
        b = small[n['name']][ay0 - oy:ay1 - oy, ax0 - ox:ax1 - ox].reshape(-1, 3).mean(0)
        pairs.append((t['name'], n['name'], a, b))
gain = {t['name']: np.ones(3) for t in tiles}
for it in range(60):
    acc = {k: [] for k in gain}
    for a_n, b_n, a, b in pairs:
        ra = a * gain[a_n]; rb = b * gain[b_n]; target = (ra + rb) / 2
        acc[a_n].append(target / np.maximum(a, 1)); acc[b_n].append(target / np.maximum(b, 1))
    for k, v in acc.items():
        if v: gain[k] = 0.5 * gain[k] + 0.5 * np.mean(v, 0)
    if FIXED:
        for k in FIXED:
            if k in gain: gain[k] = np.ones(3)
    else:
        m = np.mean(list(gain.values()), 0)
        for k in gain: gain[k] /= m
for t in tiles:
    if t['name'] in FIXED: continue
    g = np.clip(1 + (gain[t['name']] - 1) * STR, 0.9, 1.12)
    im = np.asarray(Image.open(f"{idir}/{t['name']}.png").convert('RGB'), float) * g
    Image.fromarray(im.clip(0, 255).astype(np.uint8)).save(f"{odir}/{t['name']}.png")
gs = np.array(list(gain.values())); print('gain range', gs.min(0).round(3), gs.max(0).round(3))
