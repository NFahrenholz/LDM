import torch
import torch.nn.functional as F

from diffusers import LDMPipeline
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from utils import *
from config import config

def train(args):
    device = config.device

    model, autoencoder, scheduler = get_conditional_model()
    dataloader = get_dataloader()
    x_transformer = get_xtransformer()
    # Gradient Scaling helps prevent gradients with small magnitudes from flushing to zero
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

    optimizer = torch.optim.AdamW(model.parameters(), config.learning_rate)
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(dataloader) * config.num_epochs),
    )

    start = 0
    if args.model is not None:
        checkpoint = torch.load(args.model)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start = checkpoint['epoch']
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    global_step = 0
    for epoch in range(start, config.num_epochs):
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        for step, (images, cond) in enumerate(dataloader):
            # set batch size to the amount of images got by dataloader
            # to circumvent errors in last batch
            bs = images.shape[0]

            images = images.to(device)
            cond = cond.to(device)
            cond = x_transformer(cond)

            # Encode the images into the latent space
            with torch.no_grad():
                latent_space = autoencoder.encode(images).latent_dist.sample()
                latent_space = latent_space.to(device) * 0.18215

            # Sample noise to add to the images
            noise = torch.randn(latent_space.shape).to(device)
            # Sample a random timestep for each image
            timesteps = torch.randint(0, 1000, (bs,), device=device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_ls = scheduler.add_noise(latent_space, noise, timesteps)

            # Predict the noise residual
            # If AMP is enabled the forward pass is run under autocast
            with torch.autocast(device_type="cuda", dtype=config.amp_dtype, enabled=config.use_amp):
                noise_pred = model(noisy_ls, timesteps, cond, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)

            # Scales the loss and calls backward() on scaled loss to create scaled gradients
            scaler.scale(loss).backward()
            # First unscales the gradients, if not inf or NaN optimizer.step() is then called
            scaler.step(optimizer)
            # Updates the scaler for the next iteration
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr(), "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1

        if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            evaluate(epoch, autoencoder, model, scheduler, x_transformer)

        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            state = {'epoch': epoch+1,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'scaler_state_dict': scaler.state_dict()}
            torch.save(state, os.path.join(config.output_dir, f"{epoch+1}_ckpt.pt"))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Training of the Latent Diffusion Model")

    parser.add_argument('-m', '--model', action='store', help='path to a pretrained model')

    args = parser.parse_args()

    train(args)