import random

import torch

from PIL import Image, ImageDraw
from torchmetrics.image.fid import FrechetInceptionDistance

from utils import *
from config import config

def uncond(args):
    device = config.device
    if args.device is not None:
        device = args.device

    model, autoencoder, scheduler = get_conditional_model()
    x_transformer = get_xtransformer()

    if args.model is not None:
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    scheduler.set_timesteps(args.timesteps)

    # Create the initial noise to sample from
    sample_shape = (1, config.ls_channels, config.ls_size, config.ls_size)
    latents = torch.randn(sample_shape).to(device) * scheduler.init_noise_sigma

    # Get the real Image
    data = pd.read_csv(config.dataset_path)
    real_image_path = random.choice(data["Image Path"])

    # Create the unconditional condition
    uncond_cond = torch.zeros((20, 5), dtype=torch.int)
    uncond_cond[:] = torch.IntTensor([0, 0, 0, 0, 36])
    uncond_cond = uncond_cond.flatten().to(device)

    # read the image
    img = cv2.imread(real_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # get the bounding boxes
    tree = ET.parse(f"../data/BoundingBoxes/{real_image_path.split('/')[-1]}.xml")
    root = tree.getroot()

    bboxes = get_bboxes(root)

    # transform the image and the bounding boxes
    transformed = config.transform(image=img, bboxes=bboxes)

    # concat cond and uncond
    uncond_cond = x_transformer(torch.unsqueeze(uncond_cond, 0))
    for i, ts in enumerate(tqdm(scheduler.timesteps)):
        inp = scheduler.scale_model_input(latents, ts)

        with torch.no_grad():
            pred = model(inp, ts, uncond_cond).sample

        latents = scheduler.step(pred, ts, latents).prev_sample

    sample = latent_to_pil(latents, autoencoder)[0]

    draw = ImageDraw.Draw(sample)
    for bbox in transformed['bboxes']:
        draw.rectangle(bbox[:-1], outline='blue')

    grid = make_grid([sample, Image.fromarray(transformed['image'])], 1, 2)
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    grid.save(f"{test_dir}/uncond.png")

def launch_single(args):
    device = config.device
    if args.device is not None:
        device = args.device

    model, autoencoder, scheduler = get_conditional_model()
    x_transformer = get_xtransformer()

    if args.model is not None:
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    scheduler.set_timesteps(args.timesteps)

    # Create the initial noise to sample from
    sample_shape = (1, config.ls_channels, config.ls_size, config.ls_size)
    latents = torch.randn(sample_shape).to(device) * scheduler.init_noise_sigma

    # Get the real Image
    data = pd.read_csv(config.dataset_path)
    real_image_path = random.choice(data["Image Path"])

    # Create the unconditional condition
    uncond_cond = torch.zeros((20, 5), dtype=torch.int)
    uncond_cond[:] = torch.IntTensor([0, 0, 0, 0, 36])
    uncond_cond = uncond_cond.flatten().to(device)

    # read the image
    img = cv2.imread(real_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # get the bounding boxes
    tree = ET.parse(f"../data/BoundingBoxes/{real_image_path.split('/')[-1]}.xml")
    root = tree.getroot()

    bboxes = get_bboxes(root)

    # transform the image and the bounding boxes
    transformed = config.transform(image=img, bboxes=bboxes)

    # create the conditioning
    cond = uncond_cond
    cond[:len(transformed['bboxes'] * 5)] = torch.IntTensor(transformed['bboxes']).flatten()

    # concat cond and uncond
    uncond_cond = x_transformer(torch.unsqueeze(uncond_cond, 0))
    cond = x_transformer(torch.unsqueeze(cond, 0))


    for i, ts in enumerate(tqdm(scheduler.timesteps)):
        inp = scheduler.scale_model_input(latents, ts)

        with torch.no_grad():
            u = model(inp, ts, uncond_cond).sample
            t = model(inp, ts, cond, return_dict=False)[0]

        pred = u + args.guidance_scale * (t - u)

        latents = scheduler.step(pred, ts, latents).prev_sample

    sample = latent_to_pil(latents, autoencoder)[0]

    draw = ImageDraw.Draw(sample)
    for bbox in transformed['bboxes']:
        draw.rectangle(bbox[:-1], outline='blue')

    grid = make_grid([sample, Image.fromarray(transformed['image'])], 1, 2)
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    grid.save(f"{test_dir}/grid.png")

def launch_batch(args):
    device = config.device
    bs = args.batch_size

    model, autoencoder, scheduler = get_conditional_model()
    x_transformer = get_xtransformer()

    if args.model is not None:
        checkpoint = torch.load(args.model, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']

    scheduler.set_timesteps(args.timesteps)

    # Create the initial noise to sample from
    sample_shape = (bs, config.ls_channels, config.ls_size, config.ls_size)
    latents = torch.randn(sample_shape).to(device) * scheduler.init_noise_sigma

    # Get the paths of real images
    data = pd.read_csv(config.dataset_path)
    real_images_paths = random.choices(data["Image Path"], k=config.eval_bs)
    real_images = []
    cond = torch.zeros((bs, 20, 5), dtype=torch.int)
    cond[:, :] = torch.IntTensor([0, 0, 0, 0, 36])
    cond = cond.reshape((bs, 100))
    all_bboxes = []
    for i, path in enumerate(real_images_paths):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tree = ET.parse(f"../data/BoundingBoxes/{path.split('/')[-1]}.xml")
        root = tree.getroot()

        bboxes = get_bboxes(root)

        transformed = config.transform(image=img, bboxes=bboxes)
        all_bboxes.append(transformed['bboxes'])

        real_images.append(transformed['image'])

        cond[i, :len(transformed['bboxes'] * 5)] = torch.IntTensor(transformed['bboxes']).flatten()

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

    # Draw the bboxes
    for bboxes, image in zip(all_bboxes, imgs):
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            draw.rectangle(bbox[:-1], outline='blue')

    # Make a grid of the sampled images and save them
    i = int(bs**0.5)
    imgs_grid = make_grid(imgs, rows=i, cols=i)
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    imgs_grid.save(f"{test_dir}/grid.png")

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

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Visualizing")

    parser.add_argument('-m', '--model', action='store')
    parser.add_argument('-bs', '--batch_size', action='store', default=16, type=int)
    parser.add_argument('-t', '--timesteps', action='store', default=50, type=int)
    parser.add_argument('-gs', '--guidance_scale', action='store', default=7.5, type=float)
    parser.add_argument('-d', '--device', action='store')
    parser.add_argument('-u', '--uncond', action='store', default=False)

    args = parser.parse_args()

    if args.uncond:
        uncond(args)
    else:
        launch_single(args)