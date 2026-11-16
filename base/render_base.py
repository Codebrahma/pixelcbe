"""Draw an isometric base map of a Coimbatore area from Overture buildings + OSM ground features.

Usage: python render_base.py bld.parquet osm.json out.png [px_per_m]
Heights: Overture rarely has them for Coimbatore, so they are estimated from footprint area
(with a stable per-building jitter), unless height / num_floors is present.
"""
import sys, json, math, hashlib
import pyarrow.parquet as pq
from shapely import wkt
from shapely.geometry import Polygon, LineString
from PIL import Image, ImageDraw

bld_path, osm_path, out_path = sys.argv[1:4]
S = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0  # pixels per metre
LAT0 = 11.01
MX = 111320 * math.cos(math.radians(LAT0)); MY = 110540
C30, S30 = math.cos(math.radians(30)), math.sin(math.radians(30))

def en(lon, lat):
    return (lon - 76.955) * MX, (lat - LAT0) * MY

def iso(e, n, h=0.0):
    # 2:1-ish dimetric view, looking from the south-west toward the north-east
    return ((e - n) * C30 * S, -(e + n) * S30 * S - h * S)

t = pq.read_table(bld_path).to_pylist()
osm = json.load(open(osm_path))["elements"]

def jitter(key):
    return int(hashlib.md5(key.encode()).hexdigest()[:6], 16) / 0xFFFFFF

blds = []
for r in t:
    g = wkt.loads(r["wkt"])
    polys = [g] if g.geom_type == "Polygon" else list(getattr(g, "geoms", []))
    for p in polys:
        ring = [en(x, y) for x, y in p.exterior.coords]
        pm = Polygon(ring)
        if not pm.is_valid or pm.area < 8:
            continue
        a = pm.area
        if r["height"]:
            h = r["height"]
        elif r["num_floors"]:
            h = r["num_floors"] * 3.3
        else:
            j = jitter(r["id"])
            floors = 1 + (a > 60) + (a > 150) * (j > 0.4) + (a > 400) + (a > 900) * (j > 0.5) + (a > 2000) * 2
            h = floors * 3.3 + j * 1.5
        blds.append((ring, h, r["id"]))

pts = [iso(e, n) for ring, h, _ in blds for e, n in ring]
xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
pad = 60
x0, y0 = min(xs) - pad, min(ys) - pad - 80 * S
W, H = int(max(xs) - x0 + pad), int(max(ys) - y0 + pad)
img = Image.new("RGB", (W, H), (214, 205, 180))
d = ImageDraw.Draw(img)
P = lambda e, n, h=0: (iso(e, n, h)[0] - x0, iso(e, n, h)[1] - y0)

GROUND = {"water": (104, 160, 196), "park": (150, 186, 110), "green": (164, 192, 120)}
ROADW = {"motorway": 16, "trunk": 14, "primary": 12, "secondary": 10, "tertiary": 8,
         "residential": 5, "unclassified": 5, "service": 3, "living_street": 4}
for el in osm:
    if "geometry" not in el:
        continue
    tg = el.get("tags", {})
    coords = [en(p["lon"], p["lat"]) for p in el["geometry"]]
    if tg.get("natural") == "water":
        col = GROUND["water"]
    elif tg.get("leisure") or tg.get("landuse") in ("recreation_ground",):
        col = GROUND["park"]
    elif tg.get("landuse"):
        col = GROUND["green"]
    else:
        continue
    if len(coords) > 2:
        d.polygon([P(e, n) for e, n in coords], fill=col)
for el in osm:
    tg = el.get("tags", {}); hw = tg.get("highway")
    if hw in ROADW and "geometry" in el:
        coords = [P(*en(p["lon"], p["lat"])) for p in el["geometry"]]
        d.line(coords, fill=(120, 118, 112), width=max(1, int(ROADW[hw] * S * 0.8)), joint="curve")

PALETTE = [(232, 226, 214), (226, 214, 196), (236, 232, 226), (210, 200, 186), (238, 222, 200), (220, 226, 230)]
blds.sort(key=lambda b: -max(e + n for e, n in b[0]))  # far (north-east) first
for ring, h, bid in blds:
    base = PALETTE[int(jitter(bid + "c") * len(PALETTE))]
    left = tuple(int(c * 0.78) for c in base); right = tuple(int(c * 0.62) for c in base)
    roof = tuple(min(255, int(c * 1.0)) for c in base)
    # walls: an edge is visible if its outward normal faces the viewer (south-west)
    pm = Polygon(ring); ccw = pm.exterior.is_ccw
    edges = list(zip(ring[:-1], ring[1:]))
    edges.sort(key=lambda ed: -(ed[0][0] + ed[0][1] + ed[1][0] + ed[1][1]))
    for (e1, n1), (e2, n2) in edges:
        de, dn = e2 - e1, n2 - n1
        ne, nn = (dn, -de) if ccw else (-dn, de)  # outward normal
        if ne + nn >= 0:
            continue
        col = left if ne < nn else right
        d.polygon([P(e1, n1), P(e2, n2), P(e2, n2, h), P(e1, n1, h)], fill=col)
    d.polygon([P(e, n, h) for e, n in ring], fill=roof, outline=tuple(int(c * 0.85) for c in base))

img.save(out_path)
print("buildings", len(blds), "size", img.size)
