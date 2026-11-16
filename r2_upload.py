"""Upload a directory tree to the pixelcbe-tiles R2 bucket via the Cloudflare API (parallel, skips nothing).
python r2_upload.py local_dir key_prefix [threads]"""
import os, sys, urllib.request, urllib.parse, time
from concurrent.futures import ThreadPoolExecutor

T = os.environ['full_cf_access']; A = open('/tmp/acct').read().strip()
src, prefix = sys.argv[1], sys.argv[2].strip('/')
N = int(sys.argv[3]) if len(sys.argv) > 3 else 24
CT = {'.webp': 'image/webp', '.dzi': 'application/xml', '.json': 'application/json', '.jpg': 'image/jpeg', '.png': 'image/png'}
files = []
for root, _, fs in os.walk(src):
    for f in fs:
        p = os.path.join(root, f); files.append((p, f'{prefix}/{os.path.relpath(p, src)}'))

def put(item):
    p, key = item
    url = f'https://api.cloudflare.com/client/v4/accounts/{A}/r2/buckets/pixelcbe-tiles/objects/{urllib.parse.quote(key)}'
    data = open(p, 'rb').read()
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, data=data, method='PUT', headers={
                'Authorization': f'Bearer {T}', 'Content-Type': CT.get(os.path.splitext(p)[1], 'application/octet-stream'),
                'Cache-Control': 'public, max-age=31536000, immutable'})
            urllib.request.urlopen(req, timeout=60).read(); return 0
        except Exception as e:
            time.sleep(1.5 * (attempt + 1))
    print('FAILED', key, flush=True); return 1

t0 = time.time(); fails = 0
with ThreadPoolExecutor(N) as ex:
    for i, r in enumerate(ex.map(put, files)):
        fails += r
        if i % 1000 == 0: print(i, '/', len(files), round(time.time() - t0), 's', flush=True)
print('done', len(files), 'files,', fails, 'failed,', round(time.time() - t0), 's')
