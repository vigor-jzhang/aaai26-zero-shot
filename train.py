import os
import argparse

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from utils import get_config, folder_create, setup_logger, adopt_weight
from dataloader import UNCDataloader
from model import create_vqgan_model, create_discriminator, define_optimizer, define_scheduler
from lpips import LPIPS


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default=None, help='config file path')


def create_scratch_model(ckpt_path, config):
    # define vqgan model
    vqgan = create_vqgan_model(config)
    # define discriminator
    discriminator = create_discriminator(config)
    # define optimizer
    opt_vqgan, opt_disc = define_optimizer(vqgan, discriminator, config)
    # define scheduler
    sche_vqgan, sche_disc = define_scheduler(opt_vqgan, opt_disc, config)
    # save needed modules
    torch.save({
        'epoch': 0,
        'vqgan': vqgan.state_dict(),
        'discriminator': discriminator.state_dict(),
        'opt_vqgan': opt_vqgan.state_dict(),
        'opt_disc': opt_disc.state_dict(),
        'sche_vqgan': sche_vqgan.state_dict(),
        'sche_disc': sche_disc.state_dict(),
    }, ckpt_path)
    return None


def train(previous_ckpt, config, logger):
    # define the device
    device = torch.device(config.training.device)
    # define vqgan model
    vqgan = create_vqgan_model(config)
    # define discriminator
    discriminator = create_discriminator(config)
    # move to device
    vqgan = vqgan.to(device)
    discriminator = discriminator.to(device)
    # define optimizer
    opt_vqgan, opt_disc = define_optimizer(vqgan, discriminator, config)
    # define scheduler
    sche_vqgan, sche_disc = define_scheduler(opt_vqgan, opt_disc, config)
    # load checkpoint
    checkpoint = torch.load(previous_ckpt, weights_only=True)
    vqgan.load_state_dict(checkpoint['vqgan'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    vqgan.train()
    discriminator.train()
    opt_vqgan.load_state_dict(checkpoint['opt_vqgan'])
    opt_disc.load_state_dict(checkpoint['opt_disc'])
    sche_vqgan.load_state_dict(checkpoint['sche_vqgan'])
    sche_disc.load_state_dict(checkpoint['sche_disc'])
    start_epoch = checkpoint['epoch']
    # initial dataloader
    dataloader = UNCDataloader(config)
    # define loss
    cal_perceptual_loss = LPIPS().eval().to(device=device)
    # define gradscaler
    scaler = torch.amp.GradScaler('cuda')
    # start training
    for epoch in range(start_epoch, start_epoch + config.experiment.ckpt_intervals):
        list_gan_loss = []
        list_vq_loss = []
        for (img, sobel_img, new_sobel) in dataloader:
            img, sobel_img, new_sobel = img.to(device), sobel_img.to(device), new_sobel.to(device)
            # firstly, train the discriminator
            opt_disc.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                with torch.no_grad():
                    recon_sobel_img, _ = vqgan(sobel_img)
                    recon_new_sobel, _ = vqgan(new_sobel)

                cat_gt = torch.concatenate((img, img), axis=0)
                cat_fake = torch.concatenate((recon_sobel_img, recon_new_sobel), axis=0)
                disc_real = discriminator(cat_gt)
                disc_fake = discriminator(cat_fake.detach())

                d_loss_real = torch.mean(F.relu(1. - disc_real))
                d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                disc_factor = adopt_weight(0.2, epoch, config.losses.disc_start)
                gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)
                list_gan_loss.append(gan_loss.item())
            scaler.scale(gan_loss).backward()
            scaler.step(opt_disc)
            scaler.update()
            # next is training vqgan
            opt_vqgan.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                recon_sobel_img, sobel_img_emb = vqgan(sobel_img)
                recon_new_sobel, new_sobel_emb = vqgan(new_sobel)

                cat_gt = torch.concatenate((img, img), axis=0)
                cat_fake = torch.concatenate((recon_sobel_img, recon_new_sobel), axis=0)
                
                perceptual_loss = cal_perceptual_loss(cat_gt.contiguous(), cat_fake.contiguous())
                rec_loss = torch.abs(cat_gt.contiguous() - cat_fake.contiguous())
                perceptual_rec_loss = config.losses.perceptual_factor * perceptual_loss + config.losses.recon_factor * rec_loss
                perceptual_rec_loss = perceptual_rec_loss.mean()

                disc_fake = discriminator(cat_fake)
                g_loss = -torch.mean(disc_fake)

                emb_mse_loss = F.mse_loss(sobel_img_emb, new_sobel_emb)
                emb_cos_loss = 1.0 - F.cosine_similarity(sobel_img_emb, new_sobel_emb, dim=1).mean()

                lbd = vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                vq_loss = perceptual_rec_loss + disc_factor * lbd * g_loss + 10.0 * emb_mse_loss + 5.0 * emb_cos_loss

                list_vq_loss.append(vq_loss.item())
            scaler.scale(vq_loss).backward()
            scaler.step(opt_vqgan)
            scaler.update()
        # finish one epoch
        avg_gan_loss = sum(list_gan_loss) / len(list_gan_loss)
        avg_vq_loss = sum(list_vq_loss) / len(list_vq_loss)
        print(f'Epoch {epoch} -- gan: {avg_gan_loss:.4f}, vq: {avg_vq_loss:.4f}', flush=True)
        logger.info(f'Epoch {epoch} -- gan: {avg_gan_loss:.4f}, vq: {avg_vq_loss:.4f}')
        # update the scheduler
        if epoch > config.lr_scheduler.warmup_epochs:
            sche_vqgan.step(avg_vq_loss)
        if epoch > config.lr_scheduler.warmup_epochs + config.losses.disc_start:
            sche_disc.step(avg_gan_loss)
    # save checkpoint
    checkpoint_path = config.experiment.ckpt_dir + '/ckpt_epoch{:0>6d}.pt'.format(epoch+1)
    logger.info('Save model in ['+checkpoint_path+']')
    print('Saving model in ['+checkpoint_path+']', end='', flush=True)
    torch.save({
        'epoch': epoch+1,
        'vqgan': vqgan.state_dict(),
        'discriminator': discriminator.state_dict(),
        'opt_vqgan': opt_vqgan.state_dict(),
        'opt_disc': opt_disc.state_dict(),
        'sche_vqgan': sche_vqgan.state_dict(),
        'sche_disc': sche_disc.state_dict(),
    }, checkpoint_path)
    print(' ... END', flush=True)
    return checkpoint_path


if __name__ == '__main__':
    # get config
    args = parser.parse_args()
    config = get_config(args.config_path)
    # init random seed for torch
    torch.manual_seed(config.training.seed)
    # folders create
    config = folder_create(config)
    print(config)
    # setup logger
    logger = setup_logger(config)
    # restore ckpt or create new
    if not config.experiment.restoring:
        create_scratch_model(config.experiment.ckpt_dir + '/ckpt_epoch{:0>6d}.pt'.format(0), config)
        previous_ckpt = config.experiment.ckpt_dir + '/ckpt_epoch{:0>6d}.pt'.format(0)
    else:
        previous_ckpt = config.experiment.restore_path
    # load epochs
    checkpoint = torch.load(previous_ckpt, weights_only=True)
    start_epoch = checkpoint['epoch']
    del checkpoint
    # training procedure
    for epoch in range(start_epoch, config.experiment.max_epochs, config.experiment.ckpt_intervals):
        now_ckpt = train(previous_ckpt, config, logger)
        previous_ckpt = now_ckpt
        # delete 5 intervals before checkpoint
        temp_path = config.experiment.ckpt_dir+'/ckpt_epoch{:0>6d}.pt'.format(int(epoch-5*config.experiment.ckpt_intervals))
        if os.path.exists(temp_path):
            os.remove(temp_path)
