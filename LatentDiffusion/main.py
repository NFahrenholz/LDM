from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel
from PIL import Image
from tqdm.auto import tqdm
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os

from config import config


def pil_to_latent(img):
    img = T.ToTensor()(img).unsqueeze(0) * 2.0 - 1.0
    img = img.to(device, dtype=torch.float16)
    with torch.no_grad():
        ls = vae.encode(img).latent_dist.sample() * 0.18215
    return ls


def latent_to_pil(ls):
    ls = (1 / 0.18215) * ls
    with torch.no_grad():
        img = vae.decode(ls).sample

    img = (img / 2 + 0.5).clamp(0, 1)
    img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
    imgs = (img * 255).round().astype("uint8")
    pil_images = [Image.fromarray(img) for img in imgs]
    return pil_images

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

device = config.device

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
vae.eval()

model = UNet2DModel.from_pretrained(config.unet_path).to(device)
model.eval()

scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02)

bs = config.eval_bs

scheduler.set_timesteps(config.sample_timesteps)

latents = torch.randn((bs, config.ls_channels, config.ls_size, config.ls_size)).to(device) * scheduler.init_noise_sigma

for i, ts in enumerate(tqdm(scheduler.timesteps)):
    inp = scheduler.scale_model_input(latents, ts)

    with torch.no_grad():
        noise_pred = model(inp, ts).sample

    latents = scheduler.step(noise_pred, ts, latents).prev_sample

imgs = latent_to_pil(latents)
imgs_grid = make_grid(imgs, rows=4, cols=4)

test_dir = os.path.join(config.output_dir, "samples")
os.makedirs(test_dir, exist_ok=True)
imgs_grid.save(f"sampled_{config.sample_timesteps}.png")