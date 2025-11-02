import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):

    def __init__(self):
        super(GANLoss, self).__init__()

    @staticmethod
    def disc_loss(real, fake):
        real_loss = F.mse_loss(real, torch.ones_like(real))
        fake_loss = F.mse_loss(fake, torch.zeros_like(fake))
        return real_loss, fake_loss

    @staticmethod
    def gen_loss(fake):
        gen_loss = F.mse_loss(fake, torch.ones_like(fake))
        return gen_loss
    
    @staticmethod
    def feature_loss(fmap_r, fmap_g):
        loss = 0
        # for dr, dg in zip(fmap_r, fmap_g):
        # for rl, gl in zip(dr, dg):
        for rl, gl in zip(fmap_r, fmap_g):
            loss += torch.mean(torch.abs(rl - gl))
            # loss += F.mse_loss(rl, gl)

        return loss
    
    @staticmethod
    def distill_loss(pred, ema_pred):
        loss = F.mse_loss(pred, ema_pred)
        return loss
