"""Draw the airport airfield procedurally over painted tiles (the model has never seen a runway).
Inside the aerodrome (from the base render's OSM colours): dry-grass texture, runway with markings,
taxiways with yellow centre lines, aprons. Buildings inside the airfield (terminal, hangars) keep the painting.
python airfield_fix.py tiles.json base_dir paint_dir out_dir
"""
import sys, os, json
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

names = json.load(open(sys.argv[1])); bdir, pdir, odir = sys.argv[2:5]; os.makedirs(odir, exist_ok=True)
T = 1024
near = lambda a, c, tol: np.abs(a - np.array(c, np.float32)).sum(-1) < tol
n_fixed = 0
OSM = json.load(open('../data/core_osm.json'))['elements']
STANDS = [(e['lon'], e['lat']) for e in OSM if e['type'] == 'node' and e.get('tags', {}).get('aeroway') == 'parking_position' and 'lon' in e]
STANDS += [(e['geometry'][0]['lon'], e['geometry'][0]['lat']) for e in OSM if e['type'] == 'way' and e.get('tags', {}).get('aeroway') == 'parking_position' and e.get('geometry')]
print('stands', len(STANDS))
for n in names:
    base = np.asarray(Image.open(f'{bdir}/{n}.png').convert('RGB'), np.float32)
    paint = np.asarray(Image.open(f'{pdir}/{n}.png').convert('RGB').resize((T, T)), np.float32)
    rg = base[..., 0] - base[..., 1]
    aero = near(base, (200, 192, 140), 10) & (rg > 5) & (rg < 12)
    aero = ndimage.binary_opening(aero, iterations=3)
    if aero.mean() < 0.02:
        Image.fromarray(paint.astype(np.uint8)).save(f'{odir}/{n}.png'); continue
    n_fixed += 1
    runway = near(base, (78, 78, 82), 24)
    taxi = near(base, (120, 120, 124), 18)
    apron = near(base, (170, 170, 172), 16)
    yellow = near(base, (214, 186, 60), 60)
    white = near(base, (240, 240, 236), 30) & ndimage.binary_dilation(runway, iterations=3)
    surface = runway | taxi | apron | yellow | white
    field = ndimage.binary_closing(aero | surface, iterations=6)
    field = ndimage.binary_fill_holes(field)
    field = ndimage.binary_opening(field, iterations=4)
    # buildings inside the field keep the painting
    bld = field & ~aero & ~surface
    bld = ndimage.binary_opening(bld, iterations=2)
    yy, xx = np.mgrid[0:T, 0:T]
    t = next(t for t in json.load(open(os.environ.get('PLAN', 'core/plan.json')))['tiles'] if t['name'] == n)
    gx, gy = xx + t['tx'], yy + t['ty']
    h = (((gx // 2) * 73856093) ^ ((gy // 2) * 19349663)) & 15
    lf = (np.sin(gx * 0.011 + gy * 0.007) + np.sin(gx * 0.004 - gy * 0.013) + np.sin((gx + 2 * gy) * 0.021)) / 3
    tex = np.zeros((T, T, 3), np.float32)
    tex[:] = np.array([188, 182, 132], np.float32)
    tex += (lf * 10)[..., None] * np.array([1.0, 1.1, 0.6], np.float32)
    tex += ((h - 7.5) * 0.7)[..., None]
    tufts = (h == 3) & (((gx * 7 + gy * 3) // 5) % 11 == 0)
    tex[tufts] = (150, 152, 98)
    tex[apron] = np.array([166, 166, 168], np.float32) + ((h[apron] - 7.5) * 0.6)[..., None]
    tex[taxi] = np.array([110, 110, 114], np.float32)
    tex[runway] = np.array([72, 72, 76], np.float32) + ((h[runway] - 7.5) * 0.8)[..., None]
    tex[yellow & ~runway] = (222, 190, 60)
    tex[white] = (238, 238, 232)
    # parked aircraft at OSM stands
    import math
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from iso import en, iso
    P = json.load(open('core/plan.json'))
    PLANES = []
    for (lon, lat) in STANDS:
        x, y = iso(*en(lon, lat), 0, 2.0); x = int(x - P['X0'] - t['tx']); y = int(y - P['Y0'] - t['ty'])
        if not (20 < x < T - 20 and 20 < y < T - 20) or not apron[y, x]: continue
        if all(abs(x - a) + abs(y - b) > 50 for a, b in PLANES): PLANES.append((x, y))
    if PLANES:
        from PIL import ImageDraw
        im = Image.fromarray(tex.clip(0, 255).astype(np.uint8)); dd = ImageDraw.Draw(im, 'RGBA')
        for x, y in PLANES:
            dd.polygon([(x - 30, y + 14), (x + 38, y + 14), (x + 30, y + 22), (x - 34, y + 22)], fill=(0, 0, 0, 60))      # shadow
            dd.polygon([(x + 2, y - 2), (x + 14, y - 30), (x + 22, y - 30), (x + 14, y - 2)], fill=(222, 224, 230))      # far wing
            dd.polygon([(x - 36, y - 4), (x + 30, y - 4), (x + 38, y), (x + 30, y + 4), (x - 36, y + 4), (x - 40, y)], fill=(246, 246, 248))  # fuselage
            dd.line([(x - 34, y + 1), (x + 30, y + 1)], fill=(60, 96, 170), width=2)                                       # livery stripe
            dd.polygon([(x - 2, y + 2), (x + 10, y + 2), (x - 2, y + 30), (x - 12, y + 30)], fill=(234, 236, 240))       # near wing
            dd.rectangle([x - 4, y + 10, x + 6, y + 15], fill=(150, 154, 166))                                            # engine
            dd.polygon([(x - 36, y - 4), (x - 26, y - 4), (x - 30, y - 20), (x - 38, y - 20)], fill=(60, 96, 170))        # tail fin
            dd.polygon([(x - 38, y), (x - 28, y), (x - 34, y + 9), (x - 40, y + 9)], fill=(226, 228, 234))               # tailplane
            dd.rectangle([x + 30, y - 2, x + 35, y], fill=(40, 52, 70))                                                  # cockpit
        tex = np.asarray(im, np.float32)
    m = (field & ~bld).astype(np.float32)
    m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)), np.float32)[..., None] / 255
    out = paint * (1 - m) + tex * m
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(f'{odir}/{n}.png')
print('airfield tiles', n_fixed)
