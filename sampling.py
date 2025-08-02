import os
import argparse
import random
import numpy as np
import glob
import torch

from utils import get_config
from model import create_vqgan_model
from dataloader import load_data


import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default=None, help='config file path')
parser.add_argument('--ckpt_path', type=str, default=None, help='path for checkpoint of model weights')
parser.add_argument('--res_output', type=str, default='./translation_res', help='zero-shot translation results output directory')
parser.add_argument('--trans_num', type=int, default=4, help='number of translation running')


def fetch_files(config, trans_num):
    glob_str = os.path.join(config.dataloader.data_root_dir, 'test' + config.dataloader.test_domain + '/*.npy')
    replace_key = ['/test' + config.dataloader.test_domain + '/', '/test' + config.dataloader.train_domain + '/']
    # get test domain list
    test_flist = glob.glob(glob_str)
    random.shuffle(test_flist)
    test_flist = test_flist[:trans_num]
    datadict = {}
    for item in test_flist:
        datadict[item] = item.replace(replace_key[0], replace_key[1])
    return datadict


def npy_norm(x):
    return (x - x.min())/(x.max() - x.min())


def save_png(png_path, test_img, output_img, gt_img):
    # a figure with 1 row and 3 columns
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(test_img, cmap='gray')
    axs[0].set_title("Input (Test Domain)")
    axs[0].axis("off")

    axs[1].imshow(output_img, cmap='gray')
    axs[1].set_title("Translation Output")
    axs[1].axis("off")

    axs[2].imshow(gt_img, cmap='gray')
    axs[2].set_title("GT (Training Domain)")
    axs[2].axis("off")
    plt.savefig(png_path, dpi=100)


def run_translation(config, now_ckpt, res_dir, trans_num):
    # define the device
    device = torch.device(config.training.device)
    # define vqgan model
    vqgan = create_vqgan_model(config)
    vqgan = vqgan.to(device)
    checkpoint = torch.load(now_ckpt, weights_only=True)
    vqgan.load_state_dict(checkpoint['vqgan'])
    vqgan.eval()
    # get files, dictionary:
    # key - test domain image path
    # value - path for corresponding groundtruth image in training domain
    fdict = fetch_files(config, trans_num)
    for test_path in fdict.keys():
        # get file name
        png_fn = test_path.split('/')[-1].replace('.npy', '.png')
        # get test image
        test_img, test_struct = load_data(test_path, image_size=config.vqgan.image_size, aug=False)
        # get ground truth image
        gt_path = fdict[test_path]
        gt_img, _ = load_data(gt_path, image_size=config.vqgan.image_size, aug=False)
        # run translation
        test_recon, _ = vqgan(test_struct.unsqueeze(0).to(device))
        # convert everything to npy
        test_img_npy = npy_norm(test_img.detach().cpu().squeeze().numpy())
        test_recon_npy = npy_norm(test_recon.detach().cpu().squeeze().numpy())
        gt_img_npy = npy_norm(gt_img.detach().cpu().squeeze().numpy())
        # save image
        png_path = os.path.join(res_dir, png_fn)
        save_png(png_path, test_img_npy, test_recon_npy, gt_img_npy)
    return None


if __name__ == '__main__':
    # get config
    args = parser.parse_args()
    config = get_config(args.config_path)
    now_ckpt = args.ckpt_path
    # translation results output directory
    res_dir = args.res_output
    os.makedirs(res_dir, exist_ok=True)
    # run translation
    run_translation(config, now_ckpt, res_dir, args.trans_num)
