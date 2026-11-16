"""Paint a tile grid in dependency waves so each tile continues its finished neighbours (seam fix).
python paint_wave.py tiles.json base_dir google_dir out_dir style_ref.png [workers]
tiles.json: list of {name, tx, ty} (or a plan.json with 'tiles'). Google images may be .png or .jpg.
"""
import sys, os, json, base64, time, urllib.request, io, concurrent.futures as cf
from PIL import Image

KEY = os.environ['GEMINI_API_KEY']; MODEL = os.environ.get('GEMINI_MODEL', 'gemini-3.1-flash-image')
tp, bdir, gdir, odir, ref = sys.argv[1:6]; WORKERS = int(sys.argv[6]) if len(sys.argv) > 6 else 6
d = json.load(open(tp)); tiles = d['tiles'] if isinstance(d, dict) else d
T = 1024; os.makedirs(odir, exist_ok=True)
at = {(t['tx'], t['ty']): t for t in tiles}
xs = sorted({t['tx'] for t in tiles}); ys = sorted({t['ty'] for t in tiles})
col = {x: i for i, x in enumerate(xs)}; row = {y: i for i, y in enumerate(ys)}

PROMPT = (
 "You are painting ONE tile of a giant seamless isometric pixel-art map of Coimbatore, Tamil Nadu, India.\n"
 "Image 1 = LAYOUT (isometric block render). Keep every building, road, roundabout, railway and lake exactly where it is: same footprints, heights, angle and framing. Do not add or remove streets.\n"
 "Roads in image 1 drawn raised on pillars with grey concrete girder sides are ELEVATED FLYOVERS: paint them clearly as tall concrete flyovers on round pillars, with traffic on top, the ground road and its traffic visible underneath and beside them, and ramps where they come down. NEVER flatten a flyover into a ground road, even though the aerial photo (image 2) shows it from above as a flat road; keep its pillars, its light concrete girder edge and its height exactly as in image 1. Where image 1 shows two decks at different heights, paint a two-tier flyover.\n"
 "Image 2 = real aerial photo of the same area from the same angle (buildings look flat in it). Use it only for roof colours and materials, tree canopy, open grounds, playgrounds, parking and bare earth.\n"
 "Image 3 = CONTEXT: the parts of this tile that neighbouring tiles have ALREADY painted are shown as finished pixel art; the unpainted rest is dimmed grey. "
 "Where image 3 shows finished pixel art, reproduce it exactly (same objects, colours and positions) and continue it seamlessly into the rest of the tile.\n"
 "Image 4 = style reference from this map: match its art style, palette, pixel size and detail.\n"
 "Style: detailed hand-made isometric pixel art like a modern city-builder game, crisp outlines, warm South Indian palette. "
 "Weathered flat concrete roofs with stains and parapets, black water tanks, clothes lines, small windows, shop fronts with colourful awnings and signboards on busy roads, "
 "street vendors, tea stalls, people, auto-rickshaws (yellow-black), two-wheelers, cars and buses, electric poles with wires, temple gopurams only where a temple is in the layout, "
 "coconut palms, neem and rain trees. Soft short shadows. No text, no letters, no labels, no borders.")

def b64(im):
    buf = io.BytesIO(); im.save(buf, 'PNG'); return base64.b64encode(buf.getvalue()).decode()

def gpath(name):
    for ext in ('.png', '.jpg'):
        p = f'{gdir}/{name}{ext}'
        if os.path.exists(p): return p

def context(t):
    base = Image.open(f"{bdir}/{t['name']}.png").convert('RGB')
    ctx = Image.blend(base, Image.new('RGB', base.size, (128, 128, 128)), 0.75)
    for dx in (-768, 0, 768):
        for dy in (-768, 0, 768):
            n = at.get((t['tx'] + dx, t['ty'] + dy))
            if not n or n is t or not os.path.exists(f"{odir}/{n['name']}.png"): continue
            im = Image.open(f"{odir}/{n['name']}.png").convert('RGB').resize((T, T))
            # only the half-overlap band that belongs to the neighbour (it owns the side nearer its centre)
            ox, oy = n['tx'] - t['tx'], n['ty'] - t['ty']
            x0 = max(0, ox); y0 = max(0, oy); x1 = min(T, ox + T); y1 = min(T, oy + T)
            if dx < 0: x1 = x0 + (x1 - x0) // 2 + 40
            if dx > 0: x0 = x1 - (x1 - x0) // 2 - 40
            if dy < 0: y1 = y0 + (y1 - y0) // 2 + 40
            if dy > 0: y0 = y1 - (y1 - y0) // 2 - 40
            ctx.paste(im.crop((x0 - ox, y0 - oy, x1 - ox, y1 - oy)), (x0, y0))
    return ctx

def paint(t):
    out = f"{odir}/{t['name']}.png"
    if os.path.exists(out): return t['name'], 'skip'
    imgs = [Image.open(f"{bdir}/{t['name']}.png").convert('RGB'), Image.open(gpath(t['name'])).convert('RGB'), context(t), Image.open(ref).convert('RGB')]
    parts = [{"text": PROMPT}] + [{"inline_data": {"mime_type": "image/png", "data": b64(i)}} for i in imgs]
    body = {"contents": [{"parts": parts}], "generationConfig": {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": "1:1"}}}
    for a in range(5):
        try:
            req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={KEY}", json.dumps(body).encode(), {"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=300))
            for p in r["candidates"][0]["content"]["parts"]:
                if "inlineData" in p:
                    im = Image.open(io.BytesIO(base64.b64decode(p["inlineData"]["data"]))).convert('RGB').resize((T, T))
                    import numpy as np
                    hsv = np.asarray(im.convert('HSV').resize((128, 128)), float)
                    grey = ((hsv[..., 1] < 40) & (hsv[..., 2] > 95) & (hsv[..., 2] < 165)).mean()
                    if grey > 0.45:  # copied the dimmed 'unpainted' context instead of painting
                        raise RuntimeError(f'grey copy {grey:.2f}')
                    im.save(out); return t['name'], 'ok'
        except Exception as e:
            err = str(e)[:120]
        time.sleep(6 * (a + 1))
    return t['name'], 'FAILED'

# wave index: a tile waits for left, upper-left, upper and upper-right neighbours
wave = {id(t): col[t['tx']] + 2 * row[t['ty']] for t in tiles}
ready = [t for t in tiles if gpath(t['name'])]
for w in sorted(set(wave[id(t)] for t in ready)):
    batch = [t for t in ready if wave[id(t)] == w]
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for name, st in ex.map(paint, batch): print(w, name, st, flush=True)
