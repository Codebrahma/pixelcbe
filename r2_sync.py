"""Upload a directory tree to R2 via the S3 API (fast, parallel). python r2_sync.py local_dir key_prefix [threads]"""
import os, sys, time, boto3
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor
env = dict(l.strip().split('=', 1) for l in open(os.path.expanduser('~/.secrets/r2_s3')) if '=' in l)
A = open('/tmp/acct').read().strip()
s3 = boto3.client('s3', endpoint_url=f'https://{A}.r2.cloudflarestorage.com', aws_access_key_id=env['R2_ACCESS_KEY_ID'],
                  aws_secret_access_key=env['R2_SECRET_ACCESS_KEY'], region_name='auto', config=Config(max_pool_connections=64, retries={'max_attempts': 8}))
src, prefix = sys.argv[1], sys.argv[2].strip('/'); N = int(sys.argv[3]) if len(sys.argv) > 3 else 48
CT = {'.webp': 'image/webp', '.dzi': 'application/xml', '.json': 'application/json'}
files = [(os.path.join(r, f), f'{prefix}/{os.path.relpath(os.path.join(r, f), src)}') for r, _, fs in os.walk(src) for f in fs]
def put(it):
    p, k = it
    try:
        s3.upload_file(p, 'pixelcbe-tiles', k, ExtraArgs={'ContentType': CT.get(os.path.splitext(p)[1], 'application/octet-stream'), 'CacheControl': 'public, max-age=31536000, immutable'}); return 0
    except Exception as e: print('FAILED', k, e, flush=True); return 1
t0 = time.time(); fails = 0
with ThreadPoolExecutor(N) as ex:
    for i, r in enumerate(ex.map(put, files)):
        fails += r
        if i % 2000 == 0: print(i, '/', len(files), round(time.time() - t0), 's', flush=True)
print('done', len(files), 'files,', fails, 'failed,', round(time.time() - t0), 's')
