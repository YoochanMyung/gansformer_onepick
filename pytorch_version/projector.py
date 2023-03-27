# Adapted Pbaylies Projector to Gansformer 1D Latent Vector component

import copy
import os
from time import perf_counter

import click
import numpy as np
import PIL.Image
from PIL import ImageFilter
import torch
import torch.nn.functional as F

import dnnlib
import loader

device = torch.device('cuda')

image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()

def score_images(G, model, text, latents, device, label_class = 0, batch_size = 8):
  scores = []
  all_images = []
  for i in range(latents.shape[0]//batch_size):
    images = G.synthesis(torch.tensor(latents[i*batch_size:(i+1)*batch_size,:,:], dtype=torch.float32, device=device), noise_mode='const')
    with torch.no_grad():
        image_input = (torch.clamp(images, -1, 1) + 1) * 0.5
        image_input = F.interpolate(image_input, size=(256, 256), mode='area')
        image_input = image_input[:, :, 16:240, 16:240] # 256 -> 224, center crop
        image_input -= image_mean[None, :, None, None]
        image_input /= image_std[None, :, None, None]
        score = model(image_input, text)[0]
        scores.append(score.cpu().numpy())
        all_images.append(images.cpu().numpy())

  scores = np.array(scores)
  scores = scores.reshape(-1, *scores.shape[2:]).squeeze()
  scores = 1 - scores / np.linalg.norm(scores)
  all_images = np.array(all_images)
  all_images = all_images.reshape(-1, *all_images.shape[2:])
  return scores, all_images

def project(
    G,
    target_image: torch.Tensor,
    *,
    num_steps                  = 300,
    w_avg_samples              = 8192,
    initial_learning_rate      = 0.02,
    initial_noise_factor       = 0.01,
    lr_rampdown_length         = 0.10,
    lr_rampup_length           = 0.5,
    noise_ramp_length          = 0.75,
    latent_range               = 2.0,
    max_noise                  = 0.5,
    min_threshold              = 0.6,
    use_vgg                    = True,
    use_clip                   = True,
    use_pixel                  = True,
    use_penalty                = True,
    use_center                 = True,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    device: torch.device
):
    list_loss = list()
    list_dist = list()

    if target_image is not None:
        assert target_image.shape == (G.img_channels, G.img_resolution, G.img_resolution)
    else:
        use_vgg = False
        use_pixel = False

    # reduce errors unless using clip
    if use_clip:
        import clip

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    z_samples = np.expand_dims(z_samples,axis=1)
    labels = None
    if (G.mapping.c_dim):
        labels = torch.from_numpy(0.5*np.random.RandomState(123).randn(w_avg_samples, G.mapping.c_dim)).to(device)

    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), labels)
    w_samples = w_samples.cpu().numpy().astype(np.float32)
    w_avg = np.mean(w_samples, axis=0, keepdims=True)
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    kmeans_latents = None

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    if use_vgg:
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            vgg16 = torch.jit.load(f).eval().to(device)

    # Load CLIP
    if use_clip:
        model, transform = clip.load("ViT-B/32", device=device)

    # Features for target image.
    if target_image is not None:
        target_images = target_image.unsqueeze(0).to(device).to(torch.float32)
        small_target = F.interpolate(target_images, size=(64, 64), mode='area')
        if use_center:
            center_target = F.interpolate(target_images, size=(448, 448), mode='area')[:, :, 112:336, 112:336]
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
        target_images = target_images[:, :, 16:240, 16:240] # 256 -> 224, center crop

    if use_vgg:
        vgg_target_features = vgg16(target_images, resize_images=False, return_lpips=True)
        if use_center:
            vgg_target_center = vgg16(center_target, resize_images=False, return_lpips=True)

    if use_clip:
        if target_image is not None:
            with torch.no_grad():
                clip_target_features = model.encode_image(((target_images / 255.0) - image_mean[None, :, None, None]) / image_std[None, :, None, None]).float()
                if use_center:
                    clip_target_center = model.encode_image(((center_target / 255.0) - image_mean[None, :, None, None]) / image_std[None, :, None, None]).float()

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_avg_tensor = w_opt.clone()
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = max_noise * w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = w_opt + w_noise
        synth_images = G.synthesis(torch.clamp(ws,-latent_range,latent_range), noise_mode='const')[0]

        # Downsample image to 256x256 if it's larger than that. CLIP was built for 224x224 images.
        synth_images = (torch.clamp(synth_images, -1, 1) + 1) * (255/2)
        small_synth = F.interpolate(synth_images, size=(64, 64), mode='area')

        if use_center:
            center_synth = F.interpolate(synth_images, size=(448, 448), mode='area')[:, :, 112:336, 112:336]
        synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_images = synth_images[:, :, 16:240, 16:240] # 256 -> 224, center crop

        dist = 0

        if use_vgg:
            vgg_synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
            vgg_dist =  (vgg_target_features - vgg_synth_features).square().sum()
            if use_center:
                vgg_synth_center = vgg16(center_synth, resize_images=False, return_lpips=True)
                vgg_dist += (vgg_target_center - vgg_synth_center).square().sum()
            vgg_dist *= 6
            dist += F.relu(vgg_dist*vgg_dist - min_threshold)

        if use_clip:
            clip_synth_image = ((synth_images / 255.0) - image_mean[None, :, None, None]) / image_std[None, :, None, None]
            clip_synth_features = model.encode_image(clip_synth_image).float()
            adj_center = 2.0

            if use_center:
                clip_cynth_center_image = ((center_synth / 255.0) - image_mean[None, :, None, None]) / image_std[None, :, None, None]
                adj_center = 1.0
                clip_synth_center = model.encode_image(clip_cynth_center_image).float()

            if target_image is not None:
                clip_dist =  (clip_target_features - clip_synth_features).square().sum()
                if use_center:
                    clip_dist += (clip_target_center - clip_synth_center).square().sum()
                dist += F.relu(0.5 + adj_center*clip_dist - min_threshold)

        if use_pixel:
            pixel_dist =  (target_images - synth_images).abs().sum() / 2000000.0
            if use_center:
                pixel_dist += (center_target - center_synth).abs().sum() / 2000000.0
            pixel_dist += (small_target - small_synth).square().sum() / 800000.0
            pixel_dist /= 4
            dist += F.relu(lr_ramp * pixel_dist - min_threshold)

        if use_penalty:
            l1_penalty = (w_opt - w_avg_tensor).abs().sum() / 5000.0
            dist += F.relu(lr_ramp * l1_penalty - min_threshold)

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)

        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')
        
        list_dist.append(np.round(dist.clone().detach().cpu(),2))
        list_loss.append(np.round(loss.clone().detach().cpu(),2))

        with torch.no_grad():
            torch.clamp(w_opt,-latent_range,latent_range,out=w_opt)

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()
    return w_out


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target-image', 'target_fname', help='Target image file to project to', required=False, metavar='FILE', default=None)
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
def run_projection(
    network_pkl: str,
    target_fname: str,
    outdir: str,
    num_steps: int,
):
   
    np.random.seed(1)
    torch.manual_seed(1)
    use_vgg = True
    use_clip = True
    use_pixel = True
    use_penalty = True
    use_center = True
    use_kmeans = True
    lr = 0.1
    
    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)

    G = loader.load_network(network_pkl, eval = True)["Gs"].to(device)          # Load pre-trained network

    # Load target image.
    target_image = None
    if target_fname:
        target_pil = PIL.Image.open(target_fname).convert('RGB').filter(ImageFilter.SHARPEN)
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
        target_uint8 = np.array(target_pil, dtype=np.uint8)
        target_image = torch.tensor(target_uint8.transpose([2, 0, 1]), device=device)

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project(
        G,
        target_image=target_image,
        initial_learning_rate=lr,
        num_steps=num_steps,
        use_vgg=use_vgg,
        use_clip=use_clip,
        use_pixel=use_pixel,
        use_penalty=use_penalty,
        use_center=use_center,
        device=device,
        verbose=True
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    os.makedirs(outdir, exist_ok=True)

    # Save final projected frame and W vector.
    if target_fname:
        target_pil.save(f'{outdir}/{os.path.basename(target_fname)}')
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')[0]
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/{os.path.basename(target_fname)[:-4]}_proj.png')
    np.savez(f'{outdir}/{os.path.basename(target_fname)[:-4]}_proj_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter
    
## For testing
# run_projection(["--network",'/home/ymyung/projects/src/gansformer/pytorch_version/network-snapshot-003024.pkl',"--target-image",'/home/ymyung/projects/src/gansformer/pytorch_version/cropped_jenny_1.png',"--outdir","/home/ymyung/projects/src/gansformer_onepick/pytorch_version"],standalone_mode=False)

# python projector.py --network /home/ymyung/projects/src/gansformer/pytorch_version/network-snapshot-003024.pkl --target-image /home/ymyung/projects/src/gansformer/pytorch_version/cropped_jenny_1.png --outdir ./test