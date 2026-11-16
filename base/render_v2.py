"""Chennai-style base render: extruded buildings with windows, parapets, water tanks; roads, water, parks, trees.

python render_v2.py buildings.parquet osm.json out.png S x0 y0 W H
(x0,y0,W,H) is a window of the GLOBAL iso canvas (see iso.py) at scale S, so separate renders tile together.
"""
import sys, os, json, hashlib, math
import pyarrow.parquet as pq, pyarrow as pa
from shapely import wkt
from shapely.geometry import Polygon, box
from PIL import Image, ImageDraw
from iso import en, iso, ground_at

bp, op, out = sys.argv[1:4]
S = float(sys.argv[4]); X0, Y0, W, H = map(int, sys.argv[5:9])
SS = 2  # supersample, downscaled at the end for clean edges
P = lambda e, n, h=0: ((iso(e, n, h, S)[0] - X0) * SS, (iso(e, n, h, S)[1] - Y0) * SS)

# terrain (hills rise above the flat plain): ground is drawn flat on an extended window, then draped
TER_ON = False; HG = H
if os.environ.get('TERRAIN') == '1':
    import terrain
    maxT = terrain.max_in_window(X0, Y0, W, H, S)
    if maxT > 1.0:
        TER_ON = True; HG = H + int(maxT * S) + 24
# ground-area of interest (with margin for tall buildings)
corners = [ground_at(X0 + dx, Y0 + dy, S) for dx in (-200, W + 200) for dy in (-200, HG + 400)]
aoi = Polygon([corners[0], corners[1], corners[3], corners[2]]).convex_hull

def j(key, salt=''):
    return int(hashlib.md5((key + salt).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

img = Image.new('RGB', (W * SS, HG * SS), (226, 214, 188))
GH = None
if os.environ.get('GHSL', '1') == '1' and os.path.exists(os.path.join(os.path.dirname(os.path.abspath(op)), 'ghsl_h.npy')):
    import numpy as _npg
    from iso import lonlat as _llb
    GH = _npg.load(os.path.join(os.path.dirname(os.path.abspath(op)), 'ghsl_h.npy'))
    GHT = list(map(float, open(os.path.join(os.path.dirname(os.path.abspath(op)), 'ghsl_h_transform.txt')).read().split()))
USE_WC = os.environ.get('WC') == '1'
WC_TREE_PTS = []
if USE_WC:   # ESA WorldCover 10 m land cover as the ground layer (OSM fills are drawn on top)
    import numpy as _np
    from iso import lonlat as _ll
    _wc = _np.load(os.path.join(os.path.dirname(os.path.abspath(op)), 'worldcover.npy'))
    _wx0, _wdx, _wy0, _wdy = 76.83, (77.08 - 76.83) / 3000, 11.13, -(11.13 - 10.97) / 1920
    def wc_at(e, n):
        lo, la = _ll(e, n)
        c = ((lo - _wx0) / _wdx).astype(int); r = ((la - _wy0) / _wdy).astype(int)
        ok = (c >= 0) & (c < _wc.shape[1]) & (r >= 0) & (r < _wc.shape[0])
        out = _np.full(c.shape, 50, _np.uint8); out[ok] = _wc[r[ok], c[ok]]; return out
    STEP = 4
    gy_, gx_ = _np.mgrid[0:HG + 1:STEP, 0:W + 1:STEP]
    ge, gn = ground_at(X0 + gx_.astype(float), Y0 + gy_.astype(float), S)
    cls = wc_at(ge, gn)
    COL = {10: (150, 184, 104), 20: (196, 196, 140), 30: (196, 196, 140), 40: (196, 196, 140), 50: (226, 214, 188),
           60: (226, 214, 188), 80: (226, 214, 188), 90: (150, 184, 104), 95: (150, 184, 104), 100: (196, 196, 140)}
    lut = _np.zeros((256, 3), _np.uint8); lut[:] = (226, 214, 188)
    for k_, v_ in COL.items(): lut[k_] = v_
    small = Image.fromarray(lut[cls])
    img = small.resize((W * SS, HG * SS), Image.BILINEAR)
    WC_CLS = (cls, STEP)
d = ImageDraw.Draw(img)

# ---------------- ground
osm = json.load(open(op))['elements']
def ring_of(el):
    return [en(p['lon'], p['lat']) for p in el.get('geometry', [])]
FILL = [
    (lambda t: t.get('aeroway') == 'aerodrome', (200, 192, 140)),
    (lambda t: t.get('amenity') in ('university', 'college', 'school') or t.get('landuse') == 'education', (218, 206, 174)),
    (lambda t: t.get('landuse') in ('industrial', 'railway'), (212, 204, 188)),
    (lambda t: t.get('landuse') in ('farmland', 'meadow', 'grass', 'orchard') or t.get('natural') in ('grassland', 'scrub'), (196, 196, 140)),
    (lambda t: t.get('landuse') == 'forest' or t.get('natural') == 'wood', (120, 160, 90)),
    (lambda t: t.get('leisure') in ('park', 'garden', 'playground') or t.get('landuse') in ('recreation_ground', 'cemetery'), (150, 184, 104)),
    (lambda t: t.get('leisure') in ('pitch', 'stadium'), (140, 178, 96)),
    (lambda t: t.get('natural') == 'water', (86, 146, 186)),
]
greens = []
for test, col in FILL:
    for el in osm:
        t = el.get('tags', {})
        if el['type'] == 'way' and test(t):
            r = ring_of(el)
            if len(r) > 2 and aoi.intersects(Polygon(r).buffer(0)):
                d.polygon([P(*p) for p in r], fill=col)
                if col in ((196, 196, 140), (120, 160, 90), (150, 184, 104)):
                    greens.append(Polygon(r).buffer(0))
        if el['type'] == 'relation' and test(t):
            for m in el.get('members', []):
                if m.get('role') == 'outer' and 'geometry' in m:
                    r = ring_of(m)
                    if len(r) > 2: d.polygon([P(*p) for p in r], fill=col)

# ---- airport: aprons, taxiways, runways (flat, drawn under roads)
for el in osm:
    t = el.get('tags', {})
    if el['type'] == 'way' and t.get('aeroway') == 'apron':
        r = ring_of(el)
        if len(r) > 2: d.polygon([P(*p) for p in r], fill=(170, 170, 172))
for el in osm:
    t = el.get('tags', {})
    if el['type'] == 'way' and t.get('aeroway') in ('taxiway', 'runway'):
        r = ring_of(el)
        if len(r) < 2: continue
        rw = t['aeroway'] == 'runway'
        try: wd = float(t.get('width', 45 if rw else 18))
        except ValueError: wd = 45 if rw else 18
        pts = [P(*p) for p in r]
        d.line(pts, fill=(78, 78, 82) if rw else (120, 120, 124), width=max(2, int(wd * S * SS * 0.87)), joint='curve')
        if rw:  # dashed white centreline + edge lines
            for a_, b_ in zip(r, r[1:]):
                L = math.hypot(b_[0] - a_[0], b_[1] - a_[1]); n = int(L / 30)
                for k in range(n):
                    if k % 2: continue
                    f0, f1 = k / max(n, 1), (k + 0.6) / max(n, 1)
                    q0 = (a_[0] + (b_[0] - a_[0]) * f0, a_[1] + (b_[1] - a_[1]) * f0); q1 = (a_[0] + (b_[0] - a_[0]) * f1, a_[1] + (b_[1] - a_[1]) * f1)
                    d.line([P(*q0), P(*q1)], fill=(240, 240, 236), width=max(1, int(1.2 * S * SS)))
        else:
            d.line(pts, fill=(214, 186, 60), width=max(1, int(0.6 * S * SS)))

ROADW = {'motorway': 18, 'trunk': 16, 'primary': 14, 'secondary': 12, 'tertiary': 9, 'residential': 6,
         'unclassified': 6, 'living_street': 5, 'service': 4, 'pedestrian': 4, 'track': 3, 'footway': 1.5}
roads = [(el, ROADW[el['tags']['highway']]) for el in osm if el['type'] == 'way' and el.get('tags', {}).get('highway') in ROADW]
# ---- flyovers: node heights from bridge/layer, plus ramps on connecting roads
DECK = 13.0 if os.environ.get('FLY_MASK') == '1' else 9.0; RAMP = 130.0
def _layer(t):
    try: return max(1, int(t.get('layer', '1')))
    except ValueError: return 1
zn = {}
# keep only real flyovers: connected bridge runs >= 120 m, or named flyovers/overbridges, or crossings of rail/water.
# (OSM also tags short culverts as bridge=yes; raising those creates fake ramps and loops.)
from shapely.geometry import LineString as _LS2
_br = [el for el, w in roads if el['tags'].get('bridge') in ('yes', 'viaduct') and 'nodes' in el and len(el.get('geometry', [])) > 1]
_par = {}
def _f(x):
    while _par.setdefault(x, x) != x: x = _par[x]
    return x
_nodeway = {}
for el in _br:
    for nid in el['nodes']:
        if nid in _nodeway: _par[_f(el['id'])] = _f(_nodeway[nid])
        else: _nodeway[nid] = el['id']
_len = {}
for el in _br:
    L_ = sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(ring_of(el), ring_of(el)[1:]))
    _len[_f(el['id'])] = _len.get(_f(el['id']), 0) + L_
_cross = [_LS2(ring_of(el)) for el in osm if el['type'] == 'way' and (el.get('tags', {}).get('railway') == 'rail' or el.get('tags', {}).get('waterway') in ('river', 'canal', 'stream')) and len(el.get('geometry', [])) > 1]
def _real(el):
    t = el['tags']; nm = (t.get('name', '') + ' ' + t.get('name:en', '')).lower()
    if _len.get(_f(el['id']), 0) >= 120 or 'flyover' in nm or 'overbridge' in nm or 'rob' == nm.strip(): return True
    g = _LS2(ring_of(el))
    return any(g.intersects(c) for c in _cross)
_keep = {el['id'] for el in _br if _real(el)}
for el, w in roads:
    t = el['tags']
    if t.get('bridge') in ('yes', 'viaduct') and 'nodes' in el and el['id'] not in _keep:
        t['bridge'] = 'culvert'   # treat as ground road
for el, w in roads:
    t = el['tags']
    if t.get('bridge') in ('yes', 'viaduct') and 'nodes' in el:
        for nid in el['nodes']: zn[nid] = max(zn.get(nid, 0), DECK * _layer(t))
def seglen(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])
wayz = {}
RAMP_LEN = 90.0
bridge_end_dir = {}   # node id -> (height, unit direction of the bridge arriving at that end node)
for el, w in roads:
    if el['tags'].get('bridge') in ('yes', 'viaduct') and 'nodes' in el:
        pts = ring_of(el); ids = el['nodes']
        wayz[el['id']] = [zn[i] for i in ids]
        for a_, b_, nid in ((pts[1], pts[0], ids[0]), (pts[-2], pts[-1], ids[-1])):
            dx, dy = b_[0] - a_[0], b_[1] - a_[1]; L = math.hypot(dx, dy) or 1
            bridge_end_dir[nid] = (zn[nid], (dx / L, dy / L))
for el, w in roads:
    if el['tags'].get('bridge') in ('yes', 'viaduct') or 'nodes' not in el: continue
    pts = ring_of(el); ids = el['nodes']
    z = [0.0] * len(ids)
    for order in (list(range(len(ids))), list(range(len(ids) - 1, -1, -1))):
        n0 = ids[order[0]]
        if n0 not in bridge_end_dir or len(order) < 2 or zn.get(ids[order[1]], 0) > 0: continue
        h0, (ux, uy) = bridge_end_dir[n0]
        p0, p1 = pts[order[0]], pts[order[1]]
        vx, vy = p1[0] - p0[0], p1[1] - p0[1]; L = math.hypot(vx, vy) or 1
        if (ux * vx + uy * vy) / L < 0.85:   # only a road that carries straight on (within ~30 degrees)
            continue
        dist = 0
        for k, i in enumerate(order):
            if k: dist += seglen(pts[order[k - 1]], pts[i])
            z[i] = max(z[i], h0 * max(0.0, 1 - dist / RAMP_LEN))
    if any(v > 0 for v in z): wayz[el['id']] = z
NO_FLY = os.environ.get('NO_FLY') == '1'
FLY_MASK = os.environ.get('FLY_MASK') == '1'   # draw flyover parts in marker colours for flyover_overlay.py
MK = {'shadow': (255, 255, 0), 'girder': (0, 255, 0), 'parapet': (0, 255, 255), 'deck': (255, 0, 0), 'pil_l': (0, 0, 255), 'pil_d': (255, 0, 255), 'dash': (255, 128, 0)}
GD = 4.5 if FLY_MASK else 2.8   # girder depth (m); deeper in the overlay so the deck reads as raised
def FC(kind, col): return MK[kind] if FLY_MASK else col
if NO_FLY: wayz = {}
elevated = []  # (el, w, i) segments with height > 0, drawn later in painter's order

roads.sort(key=lambda r: r[1])
def ground_runs(el):
    pts = ring_of(el); z = wayz.get(el['id'])
    if not z: return [pts]
    runs, cur = [], []
    for k in range(len(pts)):
        if z[k] <= 0.3: cur.append(pts[k])
        else:
            if len(cur) > 1: runs.append(cur)
            cur = []
    if len(cur) > 1: runs.append(cur)
    return runs
for el, w in roads:  # kerb/footpath first, asphalt on top
    for run in ground_runs(el):
        if w >= 6: d.line([P(*p) for p in run], fill=(205, 196, 178), width=int((w + 3) * S * SS * 0.87), joint='curve')
for el, w in roads:
    col = (98, 98, 102) if w >= 6 else (150, 144, 134)
    for run in ground_runs(el): d.line([P(*p) for p in run], fill=col, width=max(1, int(w * S * SS * 0.87)), joint='curve')
for el, w in roads:  # centre dashes on big roads
    if w >= 12:
        for run in ground_runs(el): d.line([P(*p) for p in run], fill=(220, 214, 190), width=max(1, int(0.4 * S * SS)))
for el, w in roads:
    z = wayz.get(el['id'])
    if not z or el['tags'].get('bridge') not in ('yes', 'viaduct'): continue
    pts = ring_of(el)
    d.line([P(p[0] + (7 if FLY_MASK else 3.5), p[1] - (7 if FLY_MASK else 3.5)) for p in pts], fill=FC('shadow', (112, 106, 96)), width=max(1, int(w * S * SS * 0.9)), joint='curve')
for el, w in roads:
    z = wayz.get(el['id'])
    if not z: continue
    pts = ring_of(el); bridge = el['tags'].get('bridge') in ('yes', 'viaduct')
    for k in range(len(pts) - 1):
        if max(z[k], z[k + 1]) > 0.3: elevated.append((pts[k], pts[k + 1], z[k], z[k + 1], w, bridge))
for el in osm:
    t = el.get('tags', {})
    if t.get('railway') == 'rail':
        pts = [P(*p) for p in ring_of(el)]
        d.line(pts, fill=(130, 118, 104), width=int(5 * S * SS)); d.line(pts, fill=(70, 66, 62), width=int(1.6 * S * SS))

# ---------------- buildings + trees, painter's order
items = []
import numpy as np, shapely
from shapely.geometry import Point
from iso import lonlat
_tab = pq.read_table(bp)
_minlon, _minlat = lonlat(*aoi.bounds[:2]); _maxlon, _maxlat = lonlat(*aoi.bounds[2:])
if 'cx' in _tab.column_names:
    cx = _tab.column('cx').to_numpy(); cy = _tab.column('cy').to_numpy()
    _tab = _tab.filter(pa.array((cx > _minlon - 0.002) & (cx < _maxlon + 0.002) & (cy > _minlat - 0.002) & (cy < _maxlat + 0.002)))
# places of worship: buildings inside these polygons (or containing these points) get a religious shape
worship = []
for el in osm:
    t = el.get('tags', {})
    if t.get('amenity') != 'place_of_worship': continue
    rel = t.get('religion', 'hindu')
    if el['type'] == 'way' and len(el.get('geometry', [])) > 2: worship.append((Polygon(ring_of(el)).buffer(1), rel))
    elif el['type'] == 'node': worship.append((Point(*en(el['lon'], el['lat'])).buffer(6), rel))
def religion_of(pm):
    for wp, rel in worship:
        if wp.intersects(pm.centroid) or wp.contains(pm.representative_point()) or (wp.area < 200 and pm.intersects(wp)): return rel
    return None
for r in _tab.to_pylist():
    g = wkt.loads(r['wkt'])
    for p in ([g] if g.geom_type == 'Polygon' else list(getattr(g, 'geoms', []))):
        ring = [en(x, y) for x, y in p.exterior.coords]
        pm = Polygon(ring)
        if not pm.is_valid or pm.area < 10 or not aoi.intersects(pm): continue
        a = pm.area; jj = j(r['id'])
        if r['height']: h = r['height']
        elif r['num_floors']: h = r['num_floors'] * 3.2
        else:
            fl = 1 + (a > 50) + (a > 110) * (jj > 0.35) + (a > 250) * (jj > 0.5) + (a > 600) + (a > 1500) * (jj > 0.4) + (a > 3000)
            if GH is not None:   # GHSL average building height of this 100 m cell scales the floor count
                cx_, cy_ = pm.centroid.x, pm.centroid.y
                lo_, la_ = _llb(cx_, cy_)
                gv = GH[min(GH.shape[0] - 1, max(0, int((la_ - GHT[5]) / GHT[4]))), min(GH.shape[1] - 1, max(0, int((lo_ - GHT[2]) / GHT[0])))]
                k_ = min(1.6, max(0.8, gv / 9.0)) if gv > 0 else 1.0
                if a > 90: fl = max(1, round(fl * k_ * (0.9 + 0.2 * j(r['id'], 'h'))))
                if a > 500 and gv > 12 and j(r['id'], 't') > 0.93: fl += 2 + int(3 * j(r['id'], 'u'))   # occasional towers in dense cells
            h = fl * 3.2 + (0.6 if jj > 0.5 else 0)
        rel = religion_of(pm) if worship and aoi.contains(pm.centroid) else None
        if rel == 'hindu': h = max(h, 6)
        items.append(('b', -max(e + n for e, n in ring), ring, h, r['id'], pm, rel))

# trees: scatter in green areas and along open ground (deterministic)
import random
rnd = random.Random(7)
for gpoly in greens:
    gp = gpoly.intersection(aoi)
    if gp.is_empty: continue
    nt = int(gp.area / 180)
    minx, miny, maxx, maxy = gp.bounds
    k = 0
    while k < nt * 3 and nt:
        e, n = rnd.uniform(minx, maxx), rnd.uniform(miny, maxy); k += 1
        from shapely.geometry import Point
        if gp.contains(Point(e, n)): items.append(('t', -(e + n), (e, n), rnd.uniform(5, 10), rnd.random(), None))

if USE_WC:
    from shapely.strtree import STRtree
    from shapely.geometry import Point as _Pt, LineString as _LS
    bpolys = [it[5] for it in items if it[0] == 'b']
    btree = STRtree(bpolys) if bpolys else None
    rlines = [_LS(ring_of(el)) for el, w in roads if len(ring_of(el)) > 1]
    rtree = STRtree(rlines) if rlines else None
    minx, miny, maxx, maxy = aoi.bounds
    rr = random.Random(int(X0 * 7 + Y0 * 13))
    # sample ground on a 4 m grid of candidate points
    cand = [(minx + i * 4 + rr.uniform(0, 4), miny + k * 4 + rr.uniform(0, 4)) for i in range(int((maxx - minx) / 4)) for k in range(int((maxy - miny) / 4))]
    import numpy as _np
    ce = _np.array([c[0] for c in cand]); cn = _np.array([c[1] for c in cand])
    cc = wc_at(ce, cn)
    SYN = os.environ.get('SYNTH') == '1'
    taken = []
    _dbg = dict(built=0, nobld=0, noroad=0, tried=0, far=0, ovl=0, near=0, ok=0)
    for (e, n), c in zip(cand, cc):
        if c not in (10, 50): continue
        if c == 50: _dbg['built'] += 1
        pt = _Pt(e, n)
        if not aoi.contains(pt): continue
        if btree is not None and len(btree.query(pt.buffer(3 if c == 10 else 4), predicate='intersects')): continue
        near_r = rtree.query(pt.buffer(5), predicate='intersects') if rtree is not None else []
        if len(near_r): continue
        if c == 10:
            if rr.random() < 0.09: items.append(('t', -(e + n), (e, n), rr.uniform(6, 11), rr.random(), None))
        elif SYN and rr.random() < 0.09:
            _dbg['tried'] += 1
            # synthetic house aligned with the nearest road (only where land cover says built-up but no footprint exists)
            if rtree is None: continue
            idx = rtree.nearest(pt); ln = rlines[idx]
            if ln.distance(pt) > 40: _dbg['far'] += 1; continue
            q = ln.interpolate(ln.project(pt)); q2 = ln.interpolate(min(ln.length, ln.project(pt) + 1))
            ux, uy = q2.x - q.x, q2.y - q.y; L_ = math.hypot(ux, uy) or 1; ux, uy = ux / L_, uy / L_
            a_, b_ = rr.uniform(6, 10) / 2, rr.uniform(7, 12) / 2
            ring = [(e + ux * sa * a_ - uy * sb * b_, n + uy * sa * a_ + ux * sb * b_) for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            pm = Polygon(ring)
            if btree is not None and len(btree.query(pm.buffer(1.5), predicate='intersects')): continue
            if any(pm.distance(o) < 1.2 for o in taken[-80:]): continue
            if len(rtree.query(pm.buffer(2.5), predicate='intersects')): continue
            taken.append(pm); _dbg['ok'] += 1
            hh = 3.2 * (1 + (rr.random() < 0.45) + (rr.random() < 0.15))
            items.append(('b', -max(x + y for x, y in ring), ring, hh, f'syn{int(e)}_{int(n)}', pm, None))

if USE_WC and os.environ.get('WCDBG'): print('wc', _dbg)
for a, b, za, zb, w, bridge in elevated:
    L = seglen(a, b); nseg = max(1, int(L / 8))
    for q in range(nseg):
        t0, t1 = q / nseg, (q + 1) / nseg
        p0 = (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0); p1 = (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)
        z0 = za + (zb - za) * t0; z1 = za + (zb - za) * t1
        if aoi.distance(Polygon([p0, p1, (p1[0] + 0.1, p1[1])]).centroid) > 60: continue
        items.append(('f', -max(p0[0] + p0[1], p1[0] + p1[1]) - 0.5, (p0, p1, z0, z1), w, bridge, None, None))
        if bridge and min(z0, z1) > 3 and q % 2 == 0:
            items.append(('p', -(p0[0] + p0[1]) - 25, p0, min(z0, z1) - 2.8, None, None, None))
items.sort(key=lambda it: it[1])
PAL = [(236, 230, 216), (230, 218, 196), (226, 222, 214), (214, 202, 184), (238, 220, 196), (218, 222, 224),
       (232, 206, 190), (222, 214, 170), (208, 216, 200), (240, 236, 228)]
ROOF_TILE = (178, 92, 62)

def shade(c, f): return tuple(max(0, min(255, int(v * f))) for v in c)

if TER_ON:
    import numpy as _np
    V_, u_, TT_ = terrain.visible(X0, Y0, W, H, S, SS, maxT)
    E_ = (u_[None, :] + V_) / 2; N_ = (V_ - u_[None, :]) / 2
    rowf = ((-(0.5 * V_) * S - Y0) * SS).astype(int).clip(0, HG * SS - 1)
    colf = _np.broadcast_to(_np.arange(W * SS)[None, :], rowf.shape)
    arr = _np.asarray(img)
    outp = arr[rowf, colf].astype(_np.float32)
    hill = TT_ > 0.05
    if hill.any():
        sh = _np.ones(TT_.shape, _np.float32); sh[hill] = terrain.shade(E_[hill], N_[hill])
        outp *= sh[..., None]
    img = Image.fromarray(outp.clip(0, 255).astype(_np.uint8)); d = ImageDraw.Draw(img)
    P = lambda e, n, h=0: ((iso(e, n, h + float(terrain.ter(e, n)), S)[0] - X0) * SS, (iso(e, n, h + float(terrain.ter(e, n)), S)[1] - Y0) * SS)
    def _anchor(it):
        if it[0] == 'b': return min(it[2], key=lambda q: q[0] + q[1])
        if it[0] == 't': return it[2]
        if it[0] == 'f': return it[2][0]
        if it[0] == 'p': return it[2]
        return None
    keep_items = []
    for it in items:
        a_ = _anchor(it)
        if a_ is not None:
            x_, y_ = P(a_[0], a_[1])
            xi, yi = int(x_), int(y_)
            if 0 <= xi < W * SS and 0 <= yi < H * SS and V_[yi, xi] < (a_[0] + a_[1]) - 15: continue
        keep_items.append(it)
    items = keep_items
for it in items:
    if it[0] == 'f':
        (p0, p1, z0, z1), w, bridge = it[2], it[3], it[4]
        de, dn = p1[0] - p0[0], p1[1] - p0[1]; L = math.hypot(de, dn) or 1
        ox, oy = -dn / L * w * 0.5, de / L * w * 0.5   # half-width offset
        A = (p0[0] + ox, p0[1] + oy); B = (p1[0] + ox, p1[1] + oy); C = (p1[0] - ox, p1[1] - oy); D = (p0[0] - ox, p0[1] - oy)
        # pick the long side facing the viewer (south-west)
        near = (D, C) if (-ox - oy) < 0 else (A, B)
        far_ok = True
        if bridge:
            # pillars every ~24 m along the centre line
            for f in (0.0,):
                pass
            d.polygon([P(*near[0], z0 - GD), P(*near[1], z1 - GD), P(*near[1], z1), P(*near[0], z0)], fill=FC('girder', (222, 218, 208)))
            d.line([P(*near[0], z0 - GD), P(*near[1], z1 - GD)], fill=FC('pil_d', (96, 92, 88)), width=max(1, int(0.7 * S * SS)))
            d.line([P(*near[0], z0 + 0.2), P(*near[1], z1 + 0.2)], fill=FC('parapet', (250, 248, 240)), width=max(1, int(0.5 * S * SS)))
        else:
            d.polygon([P(*near[0], 0), P(*near[1], 0), P(*near[1], z1), P(*near[0], z0)], fill=FC('girder', (176, 166, 150)))
        d.polygon([P(*A, z0), P(*B, z1), P(*C, z1), P(*D, z0)], fill=FC('deck', (92, 92, 98)))
        d.line([P(*A, z0 + 0.6), P(*B, z1 + 0.6)], fill=FC('parapet', (214, 208, 196)), width=max(1, int(0.5 * S * SS)))
        d.line([P(*D, z0 + 0.6), P(*C, z1 + 0.6)], fill=FC('parapet', (214, 208, 196)), width=max(1, int(0.5 * S * SS)))
        if w >= 12: d.line([P(*p0, z0), P(*p1, z1)], fill=FC('dash', (220, 214, 190)), width=max(1, int(0.4 * S * SS)))
        continue
    if it[0] == 'p':
        (e, n), zz = it[2], it[3]
        pw = 2.0
        d.polygon([P(e - pw, n - pw, 0), P(e + pw, n - pw, 0), P(e + pw, n - pw, zz), P(e - pw, n - pw, zz)], fill=FC('pil_l', (196, 190, 180)))
        d.polygon([P(e - pw, n - pw, 0), P(e - pw, n + pw, 0), P(e - pw, n + pw, zz), P(e - pw, n - pw, zz)], fill=FC('pil_d', (140, 134, 126)))
        continue
    if it[0] == 't':
        (e, n), hh, rv = it[2], it[3], it[4]
        x, y = P(e, n); r = (2.2 + 1.6 * rv) * S * SS
        palm = rv > 0.6
        d.line([(x, y), (x, y - hh * S * SS * 0.7)], fill=(110, 84, 60), width=max(1, int(0.5 * S * SS)))
        cy = y - hh * S * SS * 0.75
        if palm:
            for ang in range(0, 360, 45):
                a = math.radians(ang); d.line([(x, cy), (x + r * 1.3 * math.cos(a), cy + r * 0.7 * math.sin(a) + r * 0.3)], fill=(70, 130, 60), width=max(1, int(0.7 * S * SS)))
        else:
            d.ellipse([x - r, cy - r, x + r, cy + r * 0.9], fill=(84, 142, 70)); d.ellipse([x - r * 0.6, cy - r * 0.8, x + r * 0.4, cy + r * 0.1], fill=(112, 168, 88))
        continue
    _, _, ring, h, bid, pm, rel = it
    tiled = j(bid, 'tile') > 0.82 and h < 7
    base = PAL[int(j(bid, 'c') * len(PAL))]
    ccw = pm.exterior.is_ccw
    edges = list(zip(ring[:-1], ring[1:]))
    edges.sort(key=lambda ed: -(ed[0][0] + ed[0][1] + ed[1][0] + ed[1][1]))
    floors = max(1, int(h / 3.2))
    for (e1, n1), (e2, n2) in edges:
        de, dn = e2 - e1, n2 - n1
        ne, nn = (dn, -de) if ccw else (-dn, de)
        if ne + nn >= 0: continue
        col = shade(base, 0.86 if ne < nn else 0.70)
        d.polygon([P(e1, n1), P(e2, n2), P(e2, n2, h), P(e1, n1, h)], fill=col)
        L = math.hypot(de, dn)
        if L < 2.5: continue
        nw = int(L / 3.0)
        for f in range(floors):
            z0, z1 = f * 3.2 + 1.0, f * 3.2 + 2.2
            for k in range(nw):
                t0 = (k + 0.35) / nw; t1 = (k + 0.65) / nw
                a0 = (e1 + de * t0, n1 + dn * t0); a1 = (e1 + de * t1, n1 + dn * t1)
                d.polygon([P(*a0, z0), P(*a1, z0), P(*a1, z1), P(*a0, z1)], fill=shade(col, 0.55))
        if floors == 1 and j(bid, 'shop') > 0.5 and L > 5:  # shop shutter / awning at street level
            aw = [(152, 60, 50), (40, 96, 150), (200, 150, 40), (60, 120, 70)][int(j(bid, 'aw') * 4)]
            d.polygon([P(e1, n1, 2.6), P(e2, n2, 2.6), P(e2, n2, 3.0), P(e1, n1, 3.0)], fill=aw)
    roof = [P(e, n, h) for e, n in ring]
    if rel:
        d.polygon(roof, fill=(236, 226, 206), outline=(150, 130, 100))
        c = pm.centroid; side = min(math.sqrt(pm.area) * 0.45, 14)
        if rel == 'hindu':   # gopuram: stacked, tapering, ochre and colourful tiers
            tiers = 5 if pm.area > 300 else 3
            cols = [(214, 150, 70), (198, 90, 60), (226, 184, 80), (90, 140, 170), (214, 150, 70)]
            for k in range(tiers):
                w = side * (1 - k / (tiers + 1)); z0 = h + k * 2.4; z1 = z0 + 2.4
                q = [(c.x - w, c.y - w), (c.x + w, c.y - w), (c.x + w, c.y + w), (c.x - w, c.y + w)]
                for (e1, n1), (e2, n2) in zip(q, q[1:] + q[:1]):
                    if (n2 - n1) - (e2 - e1) < 0 or True:
                        d.polygon([P(e1, n1, z0), P(e2, n2, z0), P(e2, n2, z1), P(e1, n1, z1)], fill=shade(cols[k % 5], 0.8))
                d.polygon([P(x, y, z1) for x, y in q], fill=cols[k % 5])
            x, y = P(c.x, c.y, h + tiers * 2.4 + 1.5); r0 = 1.2 * S * SS
            d.ellipse([x - r0, y - r0, x + r0, y + r0], fill=(230, 180, 60))
        elif rel == 'christian':  # white church with a spire and cross
            w = min(side * 0.4, 3); x, y = P(c.x, c.y, h)
            d.polygon([P(c.x - w, c.y - w, h), P(c.x + w, c.y - w, h), P(c.x + w, c.y - w, h + 9), P(c.x - w, c.y - w, h + 9)], fill=(250, 250, 246))
            tip = P(c.x, c.y, h + 15); d.polygon([P(c.x - w, c.y - w, h + 9), P(c.x + w, c.y - w, h + 9), P(c.x + w, c.y + w, h + 9), tip], fill=(120, 130, 150))
            d.line([tip, (tip[0], tip[1] - 3 * S * SS)], fill=(60, 60, 60), width=max(1, SS)); d.line([(tip[0] - S * SS, tip[1] - 2 * S * SS), (tip[0] + S * SS, tip[1] - 2 * S * SS)], fill=(60, 60, 60), width=max(1, SS))
        elif rel == 'muslim':  # green dome + minaret
            x, y = P(c.x, c.y, h); r0 = min(side * 0.6, 7) * S * SS
            d.pieslice([x - r0, y - r0 * 1.4, x + r0, y + r0 * 0.6], 180, 360, fill=(70, 150, 100))
            mx, my = P(c.x + side * 0.6, c.y - side * 0.6, h)
            d.rectangle([mx - 0.8 * S * SS, my - 14 * S * SS, mx + 0.8 * S * SS, my], fill=(240, 240, 232))
            d.ellipse([mx - 1.3 * S * SS, my - 16 * S * SS, mx + 1.3 * S * SS, my - 13 * S * SS], fill=(70, 150, 100))
        continue
    if tiled:
        d.polygon(roof, fill=ROOF_TILE, outline=shade(ROOF_TILE, 0.7))
    else:
        rc = shade(base, 0.93)
        d.polygon(roof, fill=shade(base, 1.04), outline=shade(base, 0.72))
        inner = pm.buffer(-0.5)
        if inner.geom_type == 'Polygon' and not inner.is_empty:
            d.polygon([P(e, n, h) for e, n in inner.exterior.coords], fill=rc)
        # stains
        if j(bid, 'st') > 0.5 and pm.area > 40:
            c = pm.centroid; rr = math.sqrt(pm.area) * 0.18
            x, y = P(c.x + rr * 0.3, c.y - rr * 0.2, h); s = rr * S * SS
            d.ellipse([x - s, y - s * 0.5, x + s, y + s * 0.5], fill=shade(rc, 0.88))
        # water tank(s)
        if pm.area > 30:
            nt = 1 + (pm.area > 300) + (pm.area > 900)
            for k in range(nt):
                ip = pm.buffer(-1.5)
                if ip.is_empty: break
                minx, miny, maxx, maxy = ip.bounds
                tx = minx + (maxx - minx) * j(bid, f'tx{k}'); ty = miny + (maxy - miny) * j(bid, f'ty{k}')
                from shapely.geometry import Point
                if not ip.contains(Point(tx, ty)):
                    c = ip.representative_point(); tx, ty = c.x, c.y
                rr = 0.8 * S * SS; x, y = P(tx, ty, h); th = 1.4 * S * SS
                d.rectangle([x - rr, y - th, x + rr, y], fill=(34, 34, 38))
                d.ellipse([x - rr, y - th - rr * 0.5, x + rr, y - th + rr * 0.5], fill=(58, 58, 64))
                d.ellipse([x - rr, y - rr * 0.5, x + rr, y + rr * 0.5], fill=(34, 34, 38))

img = img.resize((W, H), Image.LANCZOS)
if os.environ.get('OPEN_LAND') == '1':   # far from anything built -> farmland colour (model paints fields, not invented houses)
    from scipy import ndimage
    a = np.asarray(img).astype(np.int16)
    bgm = np.abs(a - np.array([226, 214, 188])).sum(-1) < 10
    dist = ndimage.distance_transform_edt(bgm)
    k = np.clip((dist - 60) / 30, 0, 1)[..., None]
    a = a * (1 - k) + np.array([196, 196, 140]) * k
    img = Image.fromarray(a.clip(0, 255).astype(np.uint8))
img.save(out)
print('items', len(items), 'size', img.size)
