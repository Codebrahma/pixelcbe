"""Repaint one landmark in a map crop with Gemini, using a freely licensed reference photo.
python landmark_paint.py crop.png ref.jpg out.png "description of the landmark"
"""
import sys, os, json, base64, urllib.request, io
from PIL import Image
KEY = os.environ['GEMINI_API_KEY']; MODEL = os.environ.get('GEMINI_MODEL', 'gemini-3.1-flash-image')
crop, ref, out, desc = sys.argv[1:5]
def b64(p, size=None):
    im = Image.open(p).convert('RGB')
    if size: im.thumbnail(size)
    bio = io.BytesIO(); im.save(bio, 'PNG'); return base64.b64encode(bio.getvalue()).decode()
prompt = (
    "Image 1 is a tile of an isometric pixel-art city map (Pixel Coimbatore style: clean isometric view from the south-west, "
    "true 2:1 isometric angle, soft muted colours, detailed hand-pixelled buildings, small people and vehicles). "
    "Image 2 is a real photograph of the landmark that stands at the CENTRE of image 1: " + desc + ". "
    "Redraw ONLY the landmark at the centre of image 1 so it clearly and recognisably looks like the real building in image 2 "
    "(its shape, tower/gopuram, colours, steps, walls), drawn in exactly the same isometric pixel-art style, scale, camera angle and "
    "lighting as the rest of image 1. Keep its footprint at the same place and similar size. Keep REAL-WORLD SCALE: use the surrounding 2-3 storey houses as the yardstick (one storey is about 3 m); the landmark's tallest part is " + os.environ.get('SCALE', 'about 4 to 6 storeys tall') + ", never a skyscraper. The landmark must not cover neighbouring streets. Keep EVERYTHING else in image 1 "
    "unchanged: same roads, hills, trees, colours and composition, same framing. Do not add text or labels. "
    "Draw temple towers (gopurams) ONLY if the landmark itself is a Hindu temple; malls, colleges, hospitals, cinemas and offices are modern buildings without temple towers. Output one 1024x1024 image.")
parts = [{"text": prompt}, {"inline_data": {"mime_type": "image/png", "data": b64(crop)}}]
if ref == "-":
    parts[0]["text"] = prompt.replace("Image 2 is a real photograph of the landmark that stands at the CENTRE of image 1: ", "The landmark at the CENTRE of image 1 is: ").replace("looks like the real building in image 2", "looks like this real building as you know it").replace("from the photo", "")
else:
    parts.append({"inline_data": {"mime_type": "image/png", "data": b64(ref, (1024, 1024))}})
body = {"contents": [{"parts": parts}],
        "generationConfig": {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": "1:1"}}}
req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={KEY}",
                             json.dumps(body).encode(), {"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=240))
for part in r['candidates'][0]['content']['parts']:
    d = part.get('inline_data') or part.get('inlineData')
    if d:
        Image.open(io.BytesIO(base64.b64decode(d['data']))).convert('RGB').resize((1024, 1024), Image.LANCZOS).save(out); print('ok', out); break
else:
    print('no image', json.dumps(r)[:400])
