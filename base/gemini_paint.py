import sys, os, base64, json, urllib.request
key=os.environ['GEMINI_API_KEY']; model=sys.argv[1]; src=sys.argv[2]; dst=sys.argv[3]
prompt=("Redraw this isometric 3D render of an Indian city (Coimbatore, Tamil Nadu) as SimCity 3000 style pixel art. "
"Keep every building, road and lake exactly where it is, same footprint, same viewing angle and same framing. "
"Use warm South Indian colours: off-white, cream and pastel painted concrete houses with flat roofs, some terracotta tiled roofs, "
"water tanks on rooftops, coconut and neem trees in open plots, dusty grey asphalt roads. Exclude all cast shadows. No text, no labels.")
body={"contents":[{"parts":[{"text":prompt},{"inline_data":{"mime_type":"image/png","data":base64.b64encode(open(src,'rb').read()).decode()}}]}],
      "generationConfig":{"responseModalities":["IMAGE"]}}
req=urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",json.dumps(body).encode(),{"Content-Type":"application/json"})
r=json.load(urllib.request.urlopen(req,timeout=300))
for p in r["candidates"][0]["content"]["parts"]:
    if "inlineData" in p: open(dst,'wb').write(base64.b64decode(p["inlineData"]["data"])); print("ok",dst, r.get("usageMetadata",{}).get("totalTokenCount")); break
else: print(json.dumps(r)[:800])
