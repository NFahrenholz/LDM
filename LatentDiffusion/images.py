import torch

from PIL import Image, ImageDraw
from torchmetrics.image.fid import FrechetInceptionDistance

from utils import *
from config import config

def launch(args):
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
        all_bboxes.append(bboxes)

        transformed = config.transform(image=img, bboxes=bboxes)

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

    args = parser.parse_args()

    launch(args)