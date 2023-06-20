from dataclasses import dataclass

@dataclass
class TrainingConfig:
    device = "cuda"

    dataset_path = "../data/DiffusionDataset/images.csv"
    output_dir = "results/"

    image_size = 256
    ls_size = 32
    ls_channels = 4

    train_bs = 1
    eval_bs = 16
    num_epochs = 100
    learning_rate = 1e-4
    lr_warmup_steps = 500

    latent_scaling_factor = 0.18215

    save_image_epochs = 25
    save_model_epochs = 25

config = TrainingConfig()