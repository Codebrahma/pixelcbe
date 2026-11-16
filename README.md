# Pixel Coimbatore

An isometric, pixel-art, zoomable map of Coimbatore — live at **[pixelcbe.com](https://pixelcbe.com)**.
About 245 km² (Marudhamalai to the airport, Saravanampatti to Eachanari) in ~1,900 painted tiles,
inspired by [isometric.nyc / isometriccities.com](https://isometriccities.com/) and Pixel Chennai.

## How it works

```
OpenStreetMap + Overture buildings + ESA WorldCover + Copernicus DEM + GHSL heights
        │  base/render_v2.py            flat-shaded isometric "base render" per 1024 px tile
        ▼
  base render ──► FLUX.2 [klein] 4B + our LoRA (train/modal_train.py, on Modal, ~1.5 s/tile)
        │                                   paints it in the pixel-art style
        ▼
  procedural fix-ups (things the painter never saw or tends to invent):
        flyover_overlay.py   raised flyovers: deck, girders, pillars, shadows (from marker-colour mask renders)
        water_fix.py         lakes, shorelines, rail bunds
        fields_fix.py        fields / scrub / campus greenery from land cover (stops invented towns)
        hill_fix.py          forested hills from the DEM (Marudhamalai)
        airfield_fix.py      runway, aprons, aircraft at OSM stands
        sports_fix.py        stadium tracks
        color_match.py / heal_match.py   tile colour harmony + an offset "heal" layer that hides seams
        ▼
  base/build_dzi.py  ──►  Deep Zoom pyramid (webp)  ──►  Cloudflare R2  ──►  viewer/ (OpenSeadragon)
```

Landmarks (`base/landmark_job.py`, `apply_landmarks.py`) are repainted individually from freely licensed
Wikimedia Commons photos with an image model, then blended into the pyramid.

### Key pieces
- `base/iso.py` — the single global isometric frame (true isometric, camera facing NE).
- `base/core/plan.json` — tile grid (names, offsets) for the whole canvas.
- `base/render_batch.py` — renders a tile list; env `PLAN`, `TILE_LIST`, `BASE_OUT`, `WC=1` (land cover),
  `SYNTH=1` (synthetic infill houses where footprints are missing), `TERRAIN=1`, `FLY_MASK=1`.
- `train/modal_train.py` — LoRA training (ostris/ai-toolkit) and the `paint` entrypoint.

### Model
The painter is a rank-64 LoRA on FLUX.2 [klein] 4B, trained on pairs of (our base render → a painted version
of that same render). Weights: **[codeanand/pixelcbe-klein4b-lora](https://huggingface.co/codeanand/pixelcbe-klein4b-lora)**.

### Cost to reproduce
LoRA training ≈ $5 on a Modal A100; painting ≈ 1.5 s/tile. CPU work (rendering, fix-ups, pyramids) runs on a laptop.

## Make one for your city
1. Download OSM (Overpass) and Overture buildings for your bbox into `data/`, plus WorldCover/DEM rasters.
2. Make a plan (`base/core/plan.json`) of 1024 px tiles overlapping by 256 px.
3. `render_batch.py` → paint with the LoRA on Modal → run the fix-ups → `build_dzi.py`.
4. Point `viewer/index.html` at your tiles; add `labels.json` / `landmarks.json`.

## Licences & attribution
- **Code:** MIT (see `LICENSE`).
- **Map data:** © OpenStreetMap contributors (ODbL) — the rendered tiles are a Produced Work and must keep OSM attribution.
  Buildings: Overture Maps. Land cover: ESA WorldCover 2021 (CC BY 4.0). Elevation: Copernicus GLO-30.
  Building heights: GHSL, European Commission JRC.
- **Landmark paintings** are derived from Wikimedia Commons photos (mostly CC BY / CC BY-SA); see `viewer/credits.html`.
- No Google Maps imagery or data is included in this repository.

Built by [Codebrahma](https://codebrahma.com) with [Brahma](https://brahma.sh): the coding agent wrote the pipeline, trained the model, painted the city and deployed it.
