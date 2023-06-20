import os.path

from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler
from torch.utils.data import DataLoader
from PIL import Image

from config import config
from dataset import Uncond_Dataset

def get_model():
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

def get_dataloader():
    dataset = Uncond_Dataset(path=config.dataset_path, image_size=config.image_size)
    dataloader = DataLoader(dataset, batch_size=config.train_bs)

    return dataloader

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid
def evaluate(epoch, pipeline):
    images = pipeline(
        batch_size= config.eval_bs,
    ).images

    image_grid = make_grid(images, rows=4, cols=4)

    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
