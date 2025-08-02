import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .decoder import Decoder
from .finite_scalar_quantization import FSQ


class VQGAN(nn.Module):
    def __init__(self, image_size, in_ch, out_ch, mid_ch, fsq_len=6, fsq_levels=[8, 8, 8, 5, 5, 5]):
        super(VQGAN, self).__init__()
        self.encoder = Encoder(in_channels=in_ch, resolution=image_size, ch=mid_ch, emb_dim=mid_ch)
        self.decoder = Decoder(out_channels=out_ch, resolution=image_size, ch=mid_ch, emb_dim=mid_ch)
        # pre and post quant conv
        self.pre_q_conv = nn.Conv2d(mid_ch, fsq_len, 1)
        self.post_q_conv = nn.Conv2d(fsq_len, mid_ch, 1)
        # codebook
        self.codebook = FSQ(fsq_levels)
    
    def forward(self, img):
        emb = self.encoder(img)
        emb_pre_q_norm = F.normalize(self.pre_q_conv(emb), dim=1)
        img_q, q_indices = self.codebook(emb_pre_q_norm)
        emb_post_q = self.post_q_conv(img_q)
        output = self.decoder(emb_post_q)
        return output, emb_pre_q_norm
    
    def encode(self, img):
        emb = self.encoder(img)
        emb_pre_q_norm = F.normalize(self.pre_q_conv(emb), dim=1)
        img_q, q_indices = self.codebook(emb_pre_q_norm)
        return img_q, q_indices
    
    def decode(self, img_q=None, q_indices=None):
        if (img_q is None) and (q_indices is None):
            raise ValueError('img_q and q_indices are both None')
        elif img_q is None:
            img_q = self.codebook.indices_to_codes(q_indices)

        emb_post_q = self.post_q_conv(img_q)
        output = self.decoder(emb_post_q)
        return output
    
    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.conv_out
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        lda = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        lda = torch.clamp(lda, 0, 1e4).detach()

        return 0.8 * lda
    
    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path), weights_only=True)


if __name__ == '__main__':
    model = VQGAN(image_size=256, in_ch=1, out_ch=1, mid_ch=128)
    x = torch.rand(4, 1, 256, 256)
    y = model(x)
    print(y.shape)