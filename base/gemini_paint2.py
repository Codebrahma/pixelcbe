import sys, os, base64, json, urllib.request
key=os.environ['GEMINI_API_KEY']; model, src, ref, dst = sys.argv[1:5]
prompt=("Image 1 is an isometric 3D block render of a neighbourhood in Coimbatore, Tamil Nadu. Image 2 is a STYLE REFERENCE only. "
"Redraw image 1 as detailed hand-made isometric pixel art in the style of image 2: weathered flat concrete rooftops with stains, black Sintex water tanks, "
"parapet walls, small windows, clothes lines on roofs, shop awnings and signboards at street level, people, auto-rickshaws, two-wheelers, buses, "
"electric poles and wires, coconut palms and neem trees, light sandy ground. Keep every building, road and water body exactly where it is in image 1, "
"same footprint, same height, same camera angle and same framing; do not copy any content from image 2. No text, no labels.")
img=lambda p:{"inline_data":{"mime_type":"image/png","data":base64.b64encode(open(p,'rb').read()).decode()}}
body={"contents":[{"parts":[{"text":prompt},img(src),img(ref)]}],"generationConfig":{"responseModalities":["IMAGE"]}}
req=urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",json.dumps(body).encode(),{"Content-Type":"application/json"})
r=json.load(urllib.request.urlopen(req,timeout=300))
for p in r["candidates"][0]["content"]["parts"]:
    if "inlineData" in p: open(dst,'wb').write(base64.b64decode(p["inlineData"]["data"])); print("ok",dst); break
else: print(json.dumps(r)[:800])
