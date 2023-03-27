# import dnnlib
import numpy as np
import PIL.Image
import torch
import argparse
import os
import loader

device = torch.device('cpu')

def w_npz_manipulation(hyperplane_path, w_path, beta, device):
    w_hyper = np.load(hyperplane_path)['W']
    w_list = np.load(hyperplane_path)['w_list']
    w = np.load(w_path)['w'][0][0]
    w_hyper = (w_hyper/np.linalg.norm(w_hyper)) * beta
    w_hyper = w_hyper.reshape(len(w_list), -1)
    if device.type == 'cuda':
        w =  w.to('cpu')
    for idx, i in enumerate(w_list):
        w[i,:] = w[i,:] + w_hyper[idx,:]
    return torch.from_numpy(w).to(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('npz')
    parser.add_argument('outdir')
    parser.add_argument('--beta',default=0)
    parser.add_argument('--hyperplane-path',default='hyperplane_pose_new.npz')

    args = parser.parse_args()

    network_pkl = args.model
    w_path = args.npz
    outdir = args.outdir
    beta = args.beta
    hyperplane_path = args.hyperplane_path

    output_fname = os.path.join(outdir,'{}_{}.png'.format(os.path.basename(w_path)[:-4],beta))

    G = loader.load_network(network_pkl, eval = True)["Gs"].to(device)
    w = w_npz_manipulation(hyperplane_path, w_path, int(beta), device)

    image = G.synthesis(w[np.newaxis][np.newaxis], noise_mode='const')[0]
    image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    PIL.Image.fromarray(image[0],'RGB').save(output_fname)


## Testing
# python npz_to_img.py /home/ymyung/projects/src/gansformer/pytorch_version/network-snapshot-003024.pkl /home/ymyung/projects/src/gansformer_onepick/pytorch_version/test/cropped_jenny_1_proj_w.npz ./test --hyperplane-path /home/ymyung/projects/src/gansformer_onepick/pytorch_version/hyperplane_pose_new.npz --beta 50 