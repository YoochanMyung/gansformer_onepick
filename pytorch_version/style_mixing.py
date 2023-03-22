# Generate images using pretrained network pickle.
import os
import numpy as np
import PIL.Image
import argparse

import torch
import loader

device = torch.device("cpu")

def w_npz_manipulation(hyperplane_path, w, beta, device):
    w_hyper = np.load(hyperplane_path)['W']
    w_list = np.load(hyperplane_path)['w_list']
    w_hyper = (w_hyper/np.linalg.norm(w_hyper)) * beta
    w_hyper = w_hyper.reshape(len(w_list), -1)
    if device.type == 'cuda':
        w =  w.to('cpu')
    for idx, i in enumerate(w_list):
        w[i,:] = w[i,:] + w_hyper[idx,:]
    return w.to(device)

# Generate images using pretrained network pickle.
def style_mix(type_of_run, model, w_vec, seeds, styles, beta, output_dir, truncation_psi, hyperplane_path, fixed_styles, save_npz):
    seed_list = seeds.split(',')
    w_vec_list = w_vec.split(',')
    style_list = [float(x) for x in styles.split(',')]
    w_dict = dict()

    print("===========")
    print(f"Running {type_of_run}...")
    print("===========")

    print("1/4 Loading networks...")
    G = loader.load_network(model, eval = True)["Gs"].to(device)          # Load pre-trained network

    print('2/4 Generating W vectors...')
    if type_of_run == 'PS': # PS
        output_dir = os.path.join(output_dir,'PS_{}'.format(os.path.basename(w_vec_list[0].split('_proj')[0])))        
        w_vec_seed_list = [1.1]
        all_seeds = list(set(w_vec_seed_list + seed_list))
        col_z = np.stack([np.random.RandomState(int(seed)).randn(G.z_dim) for seed in all_seeds[-len(seed_list):]])
        col_z = np.expand_dims(col_z,axis=1)
        col_w = G.mapping(torch.from_numpy(col_z).to(device), None)
        w_avg = G.mapping.w_avg
        col_w = w_avg + (col_w - w_avg) * truncation_psi
        file_w = np.stack([np.load(w_vec_list[0])['w'].squeeze()])
        file_w = np.expand_dims(file_w,axis=1)
        all_w = torch.cat((torch.from_numpy(file_w).to(device), col_w), dim=0).to(device)
        w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}
    elif type_of_run == 'PP': #PP
        output_dir = os.path.join(output_dir,'PP_{}|{}'.format(os.path.basename(w_vec_list[0].split('_proj')[0]),os.path.basename(w_vec_list[1].split('_proj')[0]))) 
        w_vec_seed_list = list(np.arange(len(w_vec_list))+0.1)
        all_seeds = w_vec_seed_list
        file_w = np.stack([np.load(w)['w'].squeeze() for w in w_vec_list])
        file_w = np.expand_dims(file_w,axis=1)
        all_w = torch.from_numpy(file_w).to(device).to(device)
        w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}
    elif type_of_run == 'SS': # SS
        output_dir = os.path.join(output_dir,'SS_{}'.format('_'.join([str(x) for x in seed_list])))
        all_seeds = list(set(seed_list))
        all_seeds.sort()
        all_z = np.stack([np.random.RandomState(int(seed)).randn(G.z_dim) for seed in all_seeds])
        all_z = np.expand_dims(all_z,axis=1)
        all_w = G.mapping(torch.from_numpy(all_z).to(device), None)
        w_avg = G.mapping.w_avg
        all_w = w_avg + (all_w - w_avg) * truncation_psi
        w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}
    else:
        print("Wrong run Type")

    # Create static_styles for vector 10 to 17 from Seed 12
    if fixed_styles:
        fixed_seed = 12
        fix_col_z = np.stack([np.random.RandomState(fixed_seed).randn(G.z_dim)])
        fix_col_z = np.expand_dims(fix_col_z,axis=1)
        fix_col_w = G.mapping(torch.from_numpy(fix_col_z).to(device), None)
        w_avg = G.mapping.w_avg
        fix_col_w = w_avg + (fix_col_w - w_avg)
        static_styles = fix_col_w[0][0][range(10,18)]

    all_images = G.synthesis(all_w, noise_mode='const')[0]
    all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

    print('3/4 Generating style-mixed images...')
    if type_of_run == 'PS':
        _w_vec = w_vec_seed_list[0]
        _seed = seed_list[0]
        w = w_dict[_w_vec][0].clone() # w = Wproj, gonna be used as final w vectors
        for each_style in range(0,11):
            if not each_style == 10:
                wnproj = w[each_style]
                wnseed = w_dict[_seed][0][each_style]
                wnfinal = wnproj + (wnseed - wnproj) * style_list[each_style]                    
                w[each_style] = wnfinal
            else:
                rest_styles = list(range(10,18))
                wnproj = w[rest_styles]
                wnseed = w_dict[_seed][0][rest_styles]
                wnfinal = wnproj + (wnseed - wnproj) * style_list[each_style-1]
                if fixed_styles:
                    ## Static Styles
                    w[rest_styles] = static_styles
                else:
                    ## Unstatic Styles
                    w[rest_styles] = wnfinal

        w = w_npz_manipulation(hyperplane_path, w, int(beta), device)
        
        image = G.synthesis(w[np.newaxis][np.newaxis], noise_mode='const')[0]
        image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        image_dict[(_w_vec, _seed)] = image[0].cpu().numpy()
        
    elif type_of_run == 'SS':
        w = w_dict[seed_list[0]][0].clone() # w = Wproj, gonna be used as final w vectors
        for each_style in range(0,11):
            if not each_style == 10:
                wnproj = w[each_style]
                wnseed = w_dict[seed_list[1]][0][each_style]
                wnfinal = wnproj + (wnseed - wnproj) * style_list[each_style]                    
                w[each_style] = wnfinal
            else:
                rest_styles = list(range(10,18))
                wnproj = w[rest_styles]
                wnseed = w_dict[seed_list[1]][0][rest_styles]
                wnfinal = wnproj + (wnseed - wnproj) * style_list[each_style-1]
                if fixed_styles:
                    ## Static Styles
                    w[rest_styles] = static_styles
                else:
                    ## Unstatic Styles
                    w[rest_styles] = wnfinal

        w = w_npz_manipulation(hyperplane_path, w, int(beta), device)

        image = G.synthesis(w[np.newaxis][np.newaxis], noise_mode='const')[0]
        image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        image_dict[(seed_list[0], seed_list[1])] = image[0].cpu().numpy()

    elif type_of_run == 'PP':
        w = w_dict[w_vec_seed_list[0]][0].clone() # w = Wproj, gonna be used as final w vectors
        for each_style in range(0,11):
            if not each_style == 10:
                wnproj = w[each_style]
                wnseed = w_dict[w_vec_seed_list[1]][0][each_style]
                wnfinal = wnproj + (wnseed - wnproj) * style_list[each_style]                    
                w[each_style] = wnfinal
            else:
                rest_styles = list(range(10,18))
                wnproj = w[rest_styles]
                wnseed = w_dict[w_vec_seed_list[1]][0][rest_styles]
                wnfinal = wnproj + (wnseed - wnproj) * style_list[each_style-1]                    
                if fixed_styles:
                    ## Static Styles
                    w[rest_styles] = static_styles
                else:
                    ## Unstatic Styles
                    w[rest_styles] = wnfinal
        w = w_npz_manipulation(hyperplane_path, w, int(beta), device)

        image = G.synthesis(w[np.newaxis][np.newaxis], noise_mode='const')[0]
        image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        image_dict[(w_vec_seed_list[0], w_vec_seed_list[1])] = image[0].cpu().numpy()
  
    else:
        pass

    print('4/4 Saving images...')
    os.makedirs(output_dir, exist_ok=True)
    for key in image_dict.keys():
        if key[0] != key[1]:
            PIL.Image.fromarray(image_dict[key], 'RGB').save(f'{output_dir}/style_mixed_{key[0]}_{key[1]}.png')
            if save_npz:
                # Save W vectors
                npz_fname = os.path.join(output_dir,'{}_{}|{}|{}.npz'.format(type_of_run,key[0],key[1],'_'.join([str(x) for x in style_list])))
                np.savez(npz_fname,w=np.array(w[np.newaxis].cpu()))
#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description = "Generate images with the GANformer")
    parser.add_argument("type_of_run",          help="Determine type of run", choices=['PS','PP','SS'])
    parser.add_argument("--model",              help="Filename for a snapshot to resume", type=str)
    parser.add_argument("--seeds",              help="Comma-separated list of Seeds", type=str)
    parser.add_argument("--w_vec",              help="Projection file(s)", type=str)
    parser.add_argument("--beta",               help="Angle for shifting face", type=int)
    parser.add_argument("--styles",             help="Style ratio", default="0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5",type=str)
    parser.add_argument("--output-dir",         help="Root directory for experiments (default: %(default)s)", default="images", metavar="DIR")
    parser.add_argument("--truncation-psi",     help="Truncation Psi to be used in producing sample images (default: %(default)s)", default=0.7, type=float)
    parser.add_argument("--hyperplane-path",    help="Location of Hyperplane model")
    parser.add_argument("--fixed_styles",       help="Fix Styles for latent column 10 to 18", action='store_true')
    parser.add_argument("--save_npz",           help="Save W vectors in npz", action='store_true')

    args, _ = parser.parse_known_args()
    # For Testing #
    args.model = '/home/ymyung/projects/src/gansformer/pytorch_version/network-snapshot-003024.pkl'

    args.seeds = '3,7'
    args.w_vec = '/home/ymyung/projects/src/gansformer/pytorch_version/test_proj_w.npz,/home/ymyung/projects/src/gansformer/pytorch_version/cropped_jenny_1_proj_w.npz'
    args.hyperplane_path = '/home/ymyung/projects/src/gansformer/pytorch_version/hyperplane_pose_new.npz'
    args.beta = 0
    
    args.styles = "0.7,0.8,0.4,0.5,0.5,0.5,0.5,0.5,0.5,0.5"
    # args.styles = "0,0,0,0,0,0,0,0,0,0"
    # args.styles = "1,1,1,1,1,1,1,1,1,1"
    
    args.output_dir = '/home/ymyung/projects/src/gansformer/pytorch_version/test'
    style_mix(**vars(args))

if __name__ == "__main__":
    main()
