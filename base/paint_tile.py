"""Paint one tile with Gemini: base render (geometry) + aligned Google capture (what is really there) [+ style ref].
python paint_tile.py base.png google.png out.png [style_ref.png]
"""
import sys, os, base64, json, urllib.request, time

KEY = os.environ['GEMINI_API_KEY']
MODEL = os.environ.get('GEMINI_MODEL', 'gemini-3.1-flash-image')
base, google, out = sys.argv[1:4]
ref = sys.argv[4] if len(sys.argv) > 4 else None

PROMPT = (
    "You are painting one tile of a giant isometric pixel-art map of Coimbatore, Tamil Nadu, India.\n"
    "Image 1 is the LAYOUT: an isometric block render. Keep every building, road, roundabout, railway and water body exactly where it is, "
    "with the same footprint, height, camera angle and framing. Do not move, add or remove streets.\n"
    "Image 2 is a real aerial photo of exactly the same area from the same angle (buildings appear flat in it). Use it ONLY to decide "
    "roof colours and materials (terracotta tile, grey concrete, blue tarpaulin, metal sheet), where trees and greenery are, "
    "open grounds, playing fields, parking and bare earth. Where image 2 shows dense tree canopy, paint lush trees there.\n"
    + ("Image 3 is a finished tile of this same map: match its art style, palette, pixel size and level of detail exactly.\n" if ref else "") +
    "Style: detailed hand-made isometric pixel art like a modern city-builder game, crisp 1-2px outlines, warm South Indian palette. "
    "Weathered flat concrete roofs with stains and parapet walls, black water tanks on roofs, clothes lines, small windows on walls, "
    "shop awnings and signboards at street level, auto-rickshaws, two-wheelers, cars and buses on roads, a few people, electric poles, "
    "coconut palms, neem and rain trees, sandy light ground. Soft, short shadows only. No text, no letters, no labels, no borders."
)

def part(p):
    return {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(open(p, 'rb').read()).decode()}}

parts = [{"text": PROMPT}, part(base), part(google)] + ([part(ref)] if ref else [])
body = {"contents": [{"parts": parts}], "generationConfig": {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": "1:1", **({"imageSize": os.environ["IMG_SIZE"]} if os.environ.get("IMG_SIZE") else {})}}}
for attempt in range(4):
    try:
        req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={KEY}",
                                     json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=300))
        for p in r["candidates"][0]["content"]["parts"]:
            if "inlineData" in p:
                open(out, 'wb').write(base64.b64decode(p["inlineData"]["data"])); print("ok", out); sys.exit(0)
        print("no image", json.dumps(r)[:300])
    except Exception as e:
        print("err", e)
    time.sleep(5 * (attempt + 1))
sys.exit(1)
