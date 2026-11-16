"""Train a FLUX.2 [klein] 4B edit-LoRA on Modal with ostris/ai-toolkit:
control = our base render (OSM/Overture), target = the finished Pixel Coimbatore tile.

  modal volume put pixelcbe /tmp/ds /ds          # dataset: ds/target (*.jpg + *.txt), ds/control (*.jpg), ds/holdout
  modal run train/modal_train.py::train          # trains, saves LoRA checkpoints + samples to the volume
  modal run train/modal_train.py::paint --names c_1_2,c_3_4 --step 1500   # inference on chosen tiles

Costs are capped by the function timeouts below.
"""
import modal, os, subprocess, json

vol = modal.Volume.from_name("pixelcbe", create_if_missing=True)
hf = modal.Secret.from_name("pixelcbe-hf")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .run_commands("git clone --depth 1 https://github.com/ostris/ai-toolkit /root/ai-toolkit")
    .run_commands("cd /root/ai-toolkit && pip install --no-cache-dir torch torchvision torchaudio && pip install --no-cache-dir -r requirements.txt")
    .pip_install("hf_transfer")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "DISABLE_TELEMETRY": "YES"})
)
app = modal.App("pixelcbe-klein", image=image, volumes={"/vol": vol}, secrets=[hf])

CAP = "redraw this isometric block render of Coimbatore as detailed Pixel Coimbatore isometric pixel art"


def config(steps):
    hold = json.load(open("/vol/ds/holdout.json"))[:3]
    return {
        "job": "extension",
        "config": {
            "name": "pixelcbe_klein4b",
            "process": [{
                "type": "sd_trainer",
                "training_folder": "/vol/out",
                "device": "cuda:0",
                "network": {"type": "lora", "linear": 64, "linear_alpha": 64},
                "save": {"dtype": "bf16", "save_every": 250, "max_step_saves_to_keep": 8},
                "datasets": [{
                    "folder_path": "/vol/ds/target", "control_path": "/vol/ds/control",
                    "caption_ext": "txt", "caption_dropout_rate": 0.05,
                    "cache_latents_to_disk": True, "resolution": [1024],
                }],
                "train": {
                    "batch_size": 1, "steps": steps, "gradient_accumulation_steps": 1,
                    "train_unet": True, "train_text_encoder": False, "gradient_checkpointing": True,
                    "noise_scheduler": "flowmatch", "optimizer": "adamw8bit", "lr": 1e-4,
                    "timestep_type": "weighted", "dtype": "bf16",
                },
                "model": {"name_or_path": "black-forest-labs/FLUX.2-klein-base-4B", "arch": "flux2_klein_4b",
                          "quantize": False, "quantize_te": True},
                "sample": {
                    "sampler": "flowmatch", "sample_every": 500, "sample_start_step": 500, "width": 1024, "height": 1024,
                    "prompts": [f"{CAP} --ctrl_img /vol/ds/holdout/{h}.png" for h in hold],
                    "seed": 42, "walk_seed": False, "guidance_scale": 4, "sample_steps": 30,
                },
            }],
        },
        "meta": {"name": "pixelcbe", "version": "1.0"},
    }


@app.function(gpu="A100-80GB", timeout=3 * 3600)
def train(steps: int = 2000):
    import yaml
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    cfg = "/root/pixelcbe.yaml"
    yaml.safe_dump(config(steps), open(cfg, "w"))
    subprocess.run(["python", "run.py", cfg], cwd="/root/ai-toolkit", check=True)
    vol.commit()


@app.function(gpu="A100-80GB", timeout=3600)
def paint(names: str, step: int = 0, lora_scale: float = 1.0, steps: int = 4, model: str = "black-forest-labs/FLUX.2-klein-4B", tag: str = ""):
    """Paint tiles from /vol/ds/holdout or /vol/in with the trained LoRA. Output -> /vol/paint/<step>/<name>.png"""
    import torch, glob
    from diffusers import Flux2KleinPipeline
    from PIL import Image
    out_dir = f"/vol/out/pixelcbe_klein4b"
    final = f"{out_dir}/pixelcbe_klein4b.safetensors"
    lora = final if (not step or step >= 2000) else f"{out_dir}/pixelcbe_klein4b_{step:09d}.safetensors"
    pipe = Flux2KleinPipeline.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda")
    pipe.load_lora_weights(lora, adapter_name="px"); pipe.set_adapters(["px"], adapter_weights=[lora_scale])
    od = f"/vol/paint/{os.path.basename(lora).replace('.safetensors','')}_{model.split('/')[-1]}_{steps}st{tag}"
    os.makedirs(od, exist_ok=True)
    for n in names.split(","):
        src = next((p for p in (f"/vol/in/{n}.png", f"/vol/ds/holdout/{n}.png") if os.path.exists(p)))
        img = Image.open(src).convert("RGB")
        g = 4.0 if "base" in model else 1.0
        out = pipe(prompt=CAP, image=img, num_inference_steps=steps, guidance_scale=g, height=1024, width=1024,
                   generator=torch.Generator("cuda").manual_seed(1)).images[0]
        out.save(f"{od}/{n}.png")
    vol.commit()
    return od
