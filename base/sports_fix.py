"""Draw stadiums procedurally (athletics track, striped infield, stands) over painted tiles.
Geometry from OSM leisure=stadium polygons (>= 6000 m2). Buildings standing in front stay painted
(only pixels where the base render shows the stadium's flat fill are replaced).
python sports_fix.py plan.json base_dir paint_dir out_dir
"""
import sys, os, json, math
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iso import en, iso
from shapely.geometry import Polygon

plan = json.load(open(sys.argv[1])); bdir, pdir, odir = sys.argv[2:5]; os.makedirs(odir, exist_ok=True)
T = 1024; S = plan['S']
OSM = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/core_osm.json')))['elements']
stadia = []
for e in OSM:
    t = e.get('tags', {})
    if e['type'] == 'way' and t.get('leisure') == 'stadium' and e.get('geometry') and len(e['geometry']) > 3:
        P = Polygon([en(a['lon'], a['lat']) for a in e['geometry']]).buffer(0)
        if P.area >= 6000: stadia.append(P)
print('stadia', len(stadia))


def scr(e, n, h, t):
    x, y = iso(e, n, h, S); return (x - plan['X0'] - t['tx'], y - plan['Y0'] - t['ty'])


def oval(cx, cy, ux, uy, L, R, n=48):
    """stadium-shaped loop (straight length L, end radius R) centred at (cx,cy), long axis (ux,uy)"""
    vx, vy = -uy, ux; pts = []
    for s_, a0 in ((1, -90), (-1, 90)):
        for k in range(n + 1):
            a = math.radians(a0 + 180 * k / n)
            px = s_ * L / 2 + R * math.cos(a); py = R * math.sin(a)
            pts.append((cx + ux * px + vx * py, cy + uy * px + vy * py))
    return pts


done = 0
for t in plan['tiles']:
    n = t['name']
    if not os.path.exists(f'{pdir}/{n}.png'): continue
    box_ = Polygon([(t['tx'], t['ty']), (t['tx'] + T, t['ty']), (t['tx'] + T, t['ty'] + T), (t['tx'], t['ty'] + T)])
    hits = []
    for P in stadia:
        sp = Polygon([scr(e, n_, 0, dict(tx=0, ty=0)) for e, n_ in P.exterior.coords])
        if sp.intersects(box_): hits.append(P)
    out_p = f'{odir}/{n}.png'
    paint = Image.open(f'{pdir}/{n}.png').convert('RGB').resize((T, T))
    if not hits:
        if os.path.abspath(pdir) != os.path.abspath(odir): paint.save(out_p)
        continue
    base = np.asarray(Image.open(f'{bdir}/{n}.png').convert('RGB'), np.float32)
    flat = np.abs(base - np.array([140, 178, 96], np.float32)).sum(-1) < 30
    layer = Image.new('RGBA', (T, T), (0, 0, 0, 0)); d = ImageDraw.Draw(layer)
    for P in hits:
        r = P.minimum_rotated_rectangle; c = list(r.exterior.coords)
        e1 = (c[1][0] - c[0][0], c[1][1] - c[0][1]); e2 = (c[2][0] - c[1][0], c[2][1] - c[1][1])
        l1, l2 = math.hypot(*e1), math.hypot(*e2)
        (ux, uy), Lb, Wb = ((e1[0] / l1, e1[1] / l1), l1, l2) if l1 >= l2 else ((e2[0] / l2, e2[1] / l2), l2, l1)
        cx, cy = r.centroid.x, r.centroid.y
        # stands: whole stadium polygon in concrete, with tier lines
        d.polygon([scr(e, n_, 0, t) for e, n_ in P.exterior.coords], fill=(178, 172, 162, 255))
        for k in range(1, 5):
            Pk = P.buffer(-k * 4)
            if Pk.is_empty or Pk.geom_type != 'Polygon': break
            d.line([scr(e, n_, 0, t) for e, n_ in Pk.exterior.coords], fill=(150, 144, 136, 255), width=2)
        # track: fit inside with a margin for stands
        R = max(8.0, (Wb - 2 * 18) / 2); L = max(0.0, (Lb - 2 * 18) - 2 * R)
        d.polygon([scr(x, y, 0, t) for x, y in oval(cx, cy, ux, uy, L, R)], fill=(186, 88, 62, 255))
        for k in range(1, 7):  # lane lines
            d.line([scr(x, y, 0, t) for x, y in oval(cx, cy, ux, uy, L, R - k * 1.25)] + [scr(*oval(cx, cy, ux, uy, L, R - k * 1.25)[0], 0, t)], fill=(236, 220, 210, 255), width=1)
        Ri = R - 9.5
        if Ri > 4:
            d.polygon([scr(x, y, 0, t) for x, y in oval(cx, cy, ux, uy, L, Ri)], fill=(110, 160, 76, 255))
            # mowing stripes across the infield
            vx, vy = -uy, ux
            for k in range(-int(L / 2 + Ri), int(L / 2 + Ri), 10):
                a = (cx + ux * k, cy + uy * k); b = (cx + ux * (k + 5), cy + uy * (k + 5))
                q = [(a[0] + vx * Ri * 0.8, a[1] + vy * Ri * 0.8), (b[0] + vx * Ri * 0.8, b[1] + vy * Ri * 0.8),
                     (b[0] - vx * Ri * 0.8, b[1] - vy * Ri * 0.8), (a[0] - vx * Ri * 0.8, a[1] - vy * Ri * 0.8)]
                if abs(k) + 5 < L / 2 + Ri * 0.5:
                    d.polygon([scr(x, y, 0, t) for x, y in q], fill=(124, 174, 86, 255))
            # football pitch lines
            pl, pw = min(L + Ri, 100) / 2, min(Ri * 0.85, 34)
            rect = [(cx + ux * a_ + vx * b_, cy + uy * a_ + vy * b_) for a_, b_ in ((-pl, -pw), (pl, -pw), (pl, pw), (-pl, pw), (-pl, -pw))]
            d.line([scr(x, y, 0, t) for x, y in rect], fill=(240, 244, 236, 255), width=2)
            d.line([scr(cx + vx * pw, cy + vy * pw, 0, t), scr(cx - vx * pw, cy - vy * pw, 0, t)], fill=(240, 244, 236, 255), width=2)
    lay = np.asarray(layer, np.float32)
    a = lay[..., 3:] / 255
    out = np.asarray(paint, np.float32) * (1 - a) + lay[..., :3] * a
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(out_p); done += 1
print('stadium tiles', done)
