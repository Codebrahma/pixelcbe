"""One landmark end to end: crop from the DZI level-16 tiles, fetch a CC reference photo from Wikimedia Commons,
repaint with Gemini, and save crop/ref/out + position under /tmp/lm/<id>/.
python landmark_job.py <id> <x> <y> "<commons search>" "<description>" [ref_index]
"""
import sys, os, json, subprocess, urllib.request, urllib.parse
from PIL import Image
lid, x, y, search, desc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
ri = int(sys.argv[6]) if len(sys.argv) > 6 else 0
D = f'/tmp/lm/{lid}'; os.makedirs(D, exist_ok=True)
TILES = os.environ.get('TILES', '/home/user/workspace/viewer/tiles/v13/core_ai_files/16'); TS = 512
x0, y0 = x - 512, y - 560
im = Image.new('RGB', (1024, 1024))
for r in range(y0 // TS, (y0 + 1024) // TS + 1):
    for c in range(x0 // TS, (x0 + 1024) // TS + 1):
        f = f'{TILES}/{c}_{r}.webp'
        if os.path.exists(f): im.paste(Image.open(f), (c * TS - x0, r * TS - y0))
im.save(f'{D}/crop.png'); json.dump(dict(x0=x0, y0=y0, search=search, desc=desc), open(f'{D}/pos.json', 'w'))
UA = {'User-Agent': 'pixelcbe/0.1 (+https://github.com/Codebrahma/pixelcbe)'}
REF = os.environ.get('REF_TITLE')
if REF == '-':
    json.dump(dict(title=None), open(f'{D}/credit.json', 'w'))
    subprocess.run(['python3', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'landmark_paint.py'), f'{D}/crop.png', '-', f'{D}/out.png', desc], check=True); sys.exit(0)
if REF: search = REF
q = urllib.parse.urlencode(dict(action='query', generator='search', gsrsearch=search, gsrnamespace=6, gsrlimit=10, prop='imageinfo',
                                iiprop='url|size|extmetadata', iiurlwidth=1024, format='json'))
pages = json.load(urllib.request.urlopen(urllib.request.Request(f'https://commons.wikimedia.org/w/api.php?{q}', headers=UA)))['query']['pages']
cands = sorted(pages.values(), key=lambda p: p['index'])
cands = [p for p in cands if (REF and p['title'] == REF) or (not REF and p['imageinfo'][0]['width'] >= 800 and p['title'].lower().endswith(('.jpg', '.jpeg', '.png')))] or cands
p = cands[min(ri, len(cands) - 1)]; ii = p['imageinfo'][0]; m = ii.get('extmetadata', {})
open(f'{D}/ref.jpg', 'wb').write(urllib.request.urlopen(urllib.request.Request(ii['thumburl'], headers=UA)).read())
credit = dict(title=p['title'], url=ii['descriptionurl'], license=m.get('LicenseShortName', {}).get('value'),
              artist=m.get('Artist', {}).get('value', '')[:200])
json.dump(credit, open(f'{D}/credit.json', 'w')); print('ref', credit['title'], credit['license'])
subprocess.run(['python3', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'landmark_paint.py'), f'{D}/crop.png', f'{D}/ref.jpg', f'{D}/out.png', desc], check=True)
