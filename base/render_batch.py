"""Render many plan tiles in one process (data loaded once). python render_batch.py k n"""
import sys, json, os, runpy, functools
import pyarrow.parquet as pq
_jl = json.load; _rt = pq.read_table; _c = {}
def jl(f, *a, **k):
    key = ('j', f.name)
    if key not in _c: _c[key] = _jl(f, *a, **k)
    return _c[key]
def rt(p, *a, **k):
    if ('p', p) not in _c: _c[('p', p)] = _rt(p, *a, **k)
    return _c[('p', p)]
json.load = jl; pq.read_table = rt
k, n = int(sys.argv[1]), int(sys.argv[2])
p = _jl(open(os.environ.get('PLAN', 'core/plan.json')))
only = set(json.load(open(os.environ['TILE_LIST']))) if os.environ.get('TILE_LIST') else None
for t in p['tiles'][k::n]:
    if only is not None and t['name'] not in only: continue
    out = f"{os.environ.get('BASE_OUT', 'core/base')}/{t['name']}.png"
    if os.path.exists(out): continue
    sys.argv = ['render_v2.py', '../data/core_buildings_c.parquet', '../data/core_osm.json', out, str(p['S']), str(p['X0'] + t['tx']), str(p['Y0'] + t['ty']), '1024', '1024']
    runpy.run_path('render_v2.py', run_name='__main__')
