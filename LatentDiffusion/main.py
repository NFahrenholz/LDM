from diffusers import AutoencoderKL
from PIL import Image
import torch
import torchvision.transforms as T


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

device = "cuda"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", torch_dtype=torch.float16).to(device)
vae.eval()

img = Image.open("../data/Video01/Images/Video1_frame000090.png").convert('RGB').resize((256, 256))
ls = pil_to_latent(img)

print(ls.shape)

d_img = latent_to_pil(ls)
d_img[0].show()