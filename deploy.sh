#!/bin/bash
# Deploy the viewer to Cloudflare Pages project "pixelcbe" (pixelcbe.com, www.pixelcbe.com), then purge the CDN cache.
# Tiles live under tiles/vN/ — bump N whenever tiles change, so cached old tiles can never be served.
set -e
export CLOUDFLARE_API_TOKEN="${CLOUDFLARE_API_TOKEN:-$full_cf_access}"
Z=$(curl -s -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" "https://api.cloudflare.com/client/v4/zones?name=pixelcbe.com" | python3 -c "import json,sys;print(json.load(sys.stdin)['result'][0]['id'])")
export CLOUDFLARE_ACCOUNT_ID="${CLOUDFLARE_ACCOUNT_ID:-$(curl -s -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" "https://api.cloudflare.com/client/v4/zones/$Z" | python3 -c "import json,sys;print(json.load(sys.stdin)['result']['account']['id'])")}"
rm -rf /tmp/site && mkdir /tmp/site
cp -r viewer/{index.html,labels.json,landmarks.json,thumbs,og.jpg,credits.html} /tmp/site/
cat > /tmp/site/_headers <<'H'
/tiles/*
  Cache-Control: public, max-age=31536000, immutable
/thumbs/*
  Cache-Control: public, max-age=3600
/
  Cache-Control: no-cache
/index.html
  Cache-Control: no-cache
/*.json
  Cache-Control: no-cache
H
npx -y wrangler@3 pages deploy /tmp/site --project-name pixelcbe --branch main --commit-dirty=true
curl -s -X POST -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" -H "Content-Type: application/json" \
  "https://api.cloudflare.com/client/v4/zones/$Z/purge_cache" --data '{"purge_everything":true}' | python3 -c "import json,sys;print('cache purged:',json.load(sys.stdin)['success'])"
