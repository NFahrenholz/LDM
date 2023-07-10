import os.path
import torch
import random
import pandas as pd
import numpy as np
import cv2
import xml.etree.ElementTree as ET

from diffusers import AutoencoderKL, UNet2DModel, UNet2DConditionModel, DDPMScheduler, PNDMScheduler
from torch.utils.data import DataLoader
from PIL import Image
from tqdm.auto import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from x_transformers import TransformerWrapper, Encoder

from config import config
from dataset import Uncond_Dataset, Layout_Dataset

def get_unconditional_model():
    device = config.device

    # Load the pretrained Autoencoder
    autoencoder = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    autoencoder.eval()

    # Generate the Unet
    unet = UNet2DModel(
        sample_size=config.ls_size,
        in_channels=config.ls_channels,
        out_channels=config.ls_channels,
        layers_per_block=2,
        block_out_channels=(224, 448, 672, 896),
        down_block_types=(
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
        ),
    ).to(device)

    # Initialize the scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02)

    return unet, autoencoder, scheduler

def get_conditional_model():
    device = config.device

    # Load the pretrained Autoencoder
    autoencoder = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    autoencoder.eval()

    # Generate the Unet
    unet = UNet2DConditionModel(
        sample_size=config.ls_size,
        in_channels=config.ls_channels,
        out_channels=config.ls_channels,
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        block_out_channels=(192, 384, 576, 768),
        layers_per_block=2,
        attention_head_dim=1,
        cross_attention_dim=8192,
    ).to(device)

    # Initialize the scheduler
    scheduler = PNDMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True, steps_offset=1)

    return unet, autoencoder, scheduler

def get_dataloader():
    dataset = Layout_Dataset(path=config.dataset_path, image_size=config.image_size)
    dataloader = DataLoader(dataset, batch_size=config.train_bs, num_workers=4, pin_memory=False)

    return dataloader

def get_xtransformer():
    x_transformer = TransformerWrapper(
        num_tokens=8192,
        max_seq_len=100,
        attn_layers=Encoder(
            dim=512,
            depth=16
        )
    )

    return x_transformer.to(config.device)

def get_bboxes(root):
    bboxes = []
    for bbox in root.iter('bbox'):
        label = int(bbox.find('label_id').text)

        x1 = int(bbox.find('x1').text)
        y1 = int(bbox.find('y1').text)
        x2 = int(bbox.find('x2').text)
        y2 = int(bbox.find('y2').text)

        bboxes.append([x1, y1, x2, y2, label])

    return bboxes

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def latent_to_pil(ls, autoencoder):
    ls = (1 / 0.18215) * ls
    with torch.no_grad():
        img = autoencoder.decode(ls).sample

    img = (img / 2 + 0.5).clamp(0, 1)
    img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
    imgs = (img * 255).round().astype("uint8")
    pil_images = [Image.fromarray(img) for img in imgs]
    return pil_images

def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0

    return image

@torch.no_grad()
def evaluate(epoch, autoencoder, model, scheduler, x_transformer):
    bs = config.eval_bs
    device = config.device

    # Set the number of steps to sample
    scheduler.set_timesteps(config.sample_timesteps)

    # Create the initial noise to sample from
    sample_shape = (bs, config.ls_channels, config.ls_size, config.ls_size)
    latents = torch.randn(sample_shape).to(device) * scheduler.init_noise_sigma

    # Get the paths of real images
    data = pd.read_csv(config.dataset_path)
    real_images_paths = random.choices(data["Image Path"], k=config.eval_bs)
    real_images = []
    cond = torch.zeros((16, 20, 5), dtype=torch.int)
    cond[:, :] = torch.IntTensor([0, 0, 0, 0, 36])
    cond = cond.reshape((16, 100))
    for i, path in enumerate(real_images_paths):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tree = ET.parse(f"../data/BoundingBoxes/{path.split('/')[-1]}.xml")
        root = tree.getroot()

        bboxes = get_bboxes(root)

        transformed = config.transform(image=img, bboxes=bboxes)

        real_images.append(transformed['image'])

        cond[i, :len(transformed['bboxes']*5)] = torch.IntTensor(transformed['bboxes']).flatten()

    cond = cond.to(device)
    cond = x_transformer(cond).to(device)

    # Create the "dataset" of real images
    real_images = torch.cat([preprocess_image(image) for image in real_images])

    # Denoise the image in the set number of steps
    for i, ts in enumerate(tqdm(scheduler.timesteps)):
        # Scale the input
        inp = scheduler.scale_model_input(latents, ts)

        # Predict the noise
        with torch.no_grad():
            noise_pred = model(inp, ts, cond, return_dict=False)[0]

        # Perform a step of the scheduler
        latents = scheduler.step(noise_pred, ts, latents).prev_sample

    # Decode the latent space back into image space
    imgs = latent_to_pil(latents, autoencoder)

    # Make a grid of the sampled images and save them
    imgs_grid = make_grid(imgs, rows=4, cols=4)
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    imgs_grid.save(f"{test_dir}/{epoch + 1}.png")

    # Create the FID metric
    fid = FrechetInceptionDistance(normalize=True)

    # Create the "dataset" of fake images
    fake_images = torch.cat([preprocess_image(np.array(image)) for image in imgs])

    # update the fid
    fid.update(fake_images, real=False)
    fid.update(real_images, real=True)

    # compute the fid and print
    fid_score = float(fid.compute())
    print(f"\nFID: {fid_score}")

    return fid_score