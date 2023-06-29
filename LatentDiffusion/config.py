import torch
import albumentations as A


from albumentations.pytorch import ToTensorV2
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    device = "cuda:0"

    # paths
    dataset_path = "../data/images.csv"
    output_dir = "./results/"
    unet_path = "results/unet"

    image_size = 256
    ls_size = 32
    ls_channels = 4

    use_amp = True
    amp_dtype = torch.float16

    train_bs = 64
    eval_bs = 16
    num_epochs = 500
    learning_rate = 1e-4
    lr_warmup_steps = 1000
    lr_scheduler_name = "constant_with_warmup"

    latent_scaling_factor = 0.18215

    save_image_epochs = 10
    save_model_epochs = 50

    sample_timesteps = 50

    transform = A.Compose([
        A.Resize(image_size, image_size),
    ], bbox_params=A.BboxParams(format='pascal_voc'))

class TrainingConfigDesktop:
    device = "cuda"

    # paths
    dataset_path = "../data/DiffusionDataset/images.csv"
    output_dir = "./results/"
    unet_path = "results/unet"

    image_size = 256
    ls_size = 32
    ls_channels = 4

    use_amp = False
    amp_dtype = torch.float16

    train_bs = 1
    eval_bs = 16
    num_epochs = 500
    learning_rate = 1e-4
    lr_warmup_steps = 1000
    lr_scheduler_name = "constant_with_warmup"

    latent_scaling_factor = 0.18215

    save_image_epochs = 10
    save_model_epochs = 25

    sample_timesteps = 50

    transform = A.Compose([
        A.Resize(image_size, image_size),
    ], bbox_params=A.BboxParams(format='pascal_voc'))

config = TrainingConfigDesktop()
