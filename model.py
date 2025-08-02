import torch
from modeling import VQGAN, Discriminator
from lpips import LPIPS


def create_vqgan_model(config):
    vqgan_model = VQGAN(
        image_size=config.vqgan.image_size,
        in_ch=config.vqgan.image_channel,
        out_ch=config.vqgan.image_channel,
        mid_ch=config.vqgan.hidden_channel,
    )
    return vqgan_model


def create_discriminator(config):
    discriminator = Discriminator(image_channels=config.vqgan.image_channel)
    return discriminator


def define_optimizer(vqgan, discriminator, config):
    opt_vqgan = torch.optim.AdamW(
        vqgan.parameters(), 
        lr=config.optimizer.vqgan_learning_rate,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        eps=1e-08,
        weight_decay=config.optimizer.weight_decay
    )
    opt_disc = torch.optim.AdamW(
        discriminator.parameters(), 
        lr=config.optimizer.discriminator_learning_rate,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        eps=1e-08,
        weight_decay=config.optimizer.weight_decay
    )
    return opt_vqgan, opt_disc


def define_scheduler(opt_vqgan, opt_disc, config):
    sche_vqgan = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_vqgan,
        factor=config.lr_scheduler.factor,
        patience=config.lr_scheduler.patience,
        cooldown=config.lr_scheduler.cooldown,
        min_lr=config.lr_scheduler.min_lr,
    )
    
    sche_disc = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_disc,
        factor=config.lr_scheduler.factor,
        patience=config.lr_scheduler.patience,
        cooldown=config.lr_scheduler.cooldown,
        min_lr=config.lr_scheduler.min_lr,
    )
    return sche_vqgan, sche_disc


if __name__ == '__main__':
    from utils import get_config
    config = get_config('./configs/vqgan-config.yaml')
    
    model = create_vqgan_model(config)
    print(f"vqgan size: {sum(p.numel() for p in model.parameters())}")
    
    discriminator = create_discriminator(config)
    print(f"discriminator size: {sum(p.numel() for p in discriminator.parameters())}")

    perceptual_loss = LPIPS().eval()
    print(f"perceptual_loss size: {sum(p.numel() for p in perceptual_loss.parameters())}")
    
    x = torch.rand(4, 1, 256, 256)
    y = model(x)
    print(y.shape)