import torch
import torch.nn.functional as F

from diffusers import LDMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from utils import *
from config import config

def train():
    device = config.device

    model, autoencoder, scheduler = get_model()
    dataloader = get_dataloader()

    optimizer = torch.optim.AdamW(model.parameters(), config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(dataloader) * config.num_epochs),
    )

    global_step = 0
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        for step, images in enumerate(dataloader):
            optimizer.zero_grad()

            bs = images.shape[0]

            images = images.to(device)

            # Encode the images into the latent space
            with torch.no_grad():
                latent_space = autoencoder.encode(images).latent_dist.sample() * config.latent_scaling_factor
                latent_space = latent_space.to(device)

            # Sample noise to add to the images
            noise = torch.randn(latent_space.shape).to(device)
            # Sample a random timestep for each image
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (bs,), device=device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_ls = scheduler.add_noise(latent_space, noise, timesteps)

            # Predict the noise residual
            noise_pred = model(noisy_ls, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr(), "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1

        pipeline = LDMPipeline(vqvae=autoencoder, unet=model, scheduler=scheduler)

        if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            evaluate(epoch, pipeline)

        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            pipeline.save_pretrained(config.output_dir)

if __name__ == '__main__':
    train()