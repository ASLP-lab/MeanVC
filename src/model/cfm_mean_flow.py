import torch
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import numpy as np
from torch import nn
from src.model.utils import lens_to_mask


class Normalizer:
    # minmax for raw image, mean_std for vae latent
    def __init__(self, mode='minmax', mean=None, std=None):
        assert mode in ['minmax', 'mean_std'], "mode must be 'minmax' or 'mean_std'"
        self.mode = mode

        if mode == 'mean_std':
            if mean is None or std is None:
                raise ValueError("mean and std must be provided for 'mean_std' mode")
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

    @classmethod
    def from_list(cls, config):
        """
        config: [mode, mean, std]
        """
        mode, mean, std = config
        return cls(mode, mean, std)

    def norm(self, x):
        if self.mode == 'minmax':
            return x * 2 - 1
        elif self.mode == 'mean_std':
            return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnorm(self, x):
        if self.mode == 'minmax':
            return (x + 1) * 0.5
        elif self.mode == 'mean_std':
            return x * self.std.to(x.device) + self.mean.to(x.device)


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, mask, gamma=0., c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, D)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=-1)
    delta_sq = delta_sq[mask]
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()


class MeanFlow:
    def __init__(
        self,
        # transformer: nn.Module,
        normalizer=['minmax', None, None],
        # mean flow settings
        flow_ratio=0.50,
        # time distribution, mu, sigma
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0.10,
        cfg_scale=2.0,
        # experimental
        cfg_uncond='u',
        p=0.5,
        jvp_api='autograd',
    ):
        super().__init__()
        # self.channels = channels
        # self.image_size = image_size
        # self.num_classes = num_classes
        # self.use_cond = num_classes is not None

        self.normer = Normalizer.from_list(normalizer)

        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.cfg_ratio = cfg_ratio
        self.w = cfg_scale
        self.p = p

        self.cfg_uncond = cfg_uncond
        self.jvp_api = jvp_api

        assert jvp_api in ['funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

    # fix: r should be always not larger than t
    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid

        # Assign t = max, r = min, for each pair
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    def sample_t_r_one_step(self, batch_size, device):
        t_np = np.ones(batch_size, dtype=np.float32)
        r_np = np.zeros(batch_size, dtype=np.float32)
        
        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    # def loss(self, model, x, c=None):
    def loss(
        self,
        model,
        x:torch.Tensor,
        bn: torch.Tensor,
        spks: torch.Tensor,
        prompts: torch.Tensor,
        inputs_length: torch.tensor,
        ):
        batch_size, seq_len, _ = x.size()
        device = x.device

        t, r = self.sample_t_r(batch_size, device)

        t_ = rearrange(t, "b -> b 1 1")
        r_ = rearrange(r, "b -> b 1 1")

        e = torch.randn_like(x)
        # x = self.normer.norm(x)

        original_mask = lens_to_mask(inputs_length, length=seq_len)
        cache = x
        
        z = (1 - t_) * x + t_ * e
        v = e - x
        
        if self.w is not None:
            with torch.no_grad():
                u_t = model(z,
                        t,
                        t, 
                        cache=cache, 
                        cond=bn, 
                        spks=spks,
                        prompts=prompts,
                        mask=original_mask,
                        is_uncondition=True)
            v_hat = self.w * v + (1 - self.w) * u_t
        else:
            v_hat = v

        random_tensor = torch.rand(batch_size, device=x.device)
        cfg_mask = random_tensor < self.cfg_ratio
        # c = torch.where(cfg_mask, uncond, c)

        if self.cfg_uncond == 'v':
            # as v = wv - (1-w)v = wv - (1-w)u in the unconditional case, should we directly use v instead?
            # todo: why use r generate cfg_mask?
            cfg_mask_ = rearrange(r, "b -> b 1 1").bool()
            v_hat = torch.where(cfg_mask_, v, v_hat)

        # forward pass
        # u = model(z, t, r, y=c)
        model_partial = partial(model, cache=cache, cond=bn, spks=spks, prompts=prompts,
                               mask=original_mask,
                               cfg_mask=cfg_mask
                               )
        jvp_args = (
            lambda z, t, r: model_partial(z, t, r),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )

        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)

        u_tgt = v_hat - (t_ - r_) * dudt
        
        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error, original_mask, gamma=1 - self.p)
        # loss = adaptive_l2_loss(error[0], original_mask[0])
        # loss = F.mse_loss(u, stopgrad(u_tgt))

        mse_val = (stopgrad(error) ** 2)
        mse_val = torch.mean(mse_val, dim=-1)
        mse_val = mse_val[original_mask].mean()

        return loss, mse_val

    def loss_one_step_only(
        self,
        model,
        x:torch.Tensor,
        bn: torch.Tensor,
        spks: torch.Tensor,
        prompts: torch.Tensor,
        inputs_length: torch.tensor,
        ):
        batch_size, seq_len, _ = x.size()
        device = x.device


        t, r = self.sample_t_r_one_step(batch_size, device)

        t_ = rearrange(t, "b -> b 1 1")
        r_ = rearrange(r, "b -> b 1 1")

        e = torch.randn_like(x)
        # x = self.normer.norm(x)

        original_mask = lens_to_mask(inputs_length, length=seq_len)
        cache = x
        
        z = (1 - t_) * x + t_ * e
        v = e - x    # 数据到噪声

        u = model(
                z,
                t,
                r,
                cache=cache,
                cond=bn,
                spks=spks,
                prompts=prompts,
                mask=original_mask,
            )
        
        pred = z - u
        # error = u - stopgrad(u_tgt)
        error = u - v

        loss = adaptive_l2_loss(error, original_mask, gamma=1 - self.p)

        mse_val = (stopgrad(error) ** 2)
        mse_val = torch.mean(mse_val, dim=-1)
        mse_val = mse_val[original_mask].mean()

        return loss, mse_val, pred, original_mask
    
    
    def loss_ema_one_step_only(
        self,
        model,
        ema_model,
        x:torch.Tensor,
        bn: torch.Tensor,
        spks: torch.Tensor,
        prompts: torch.Tensor,
        inputs_length: torch.tensor,
        ):
        batch_size, seq_len, _ = x.size()
        device = x.device


        t, r = self.sample_t_r_one_step(batch_size, device)

        t_ = rearrange(t, "b -> b 1 1")
        r_ = rearrange(r, "b -> b 1 1")

        e = torch.randn_like(x)
        # x = self.normer.norm(x)

        original_mask = lens_to_mask(inputs_length, length=seq_len)
        cache = x
        
        z = (1 - t_) * x + t_ * e
        v = e - x

        u = model(
                z,
                t,
                r,
                cache=cache,
                cond=bn,
                spks=spks,
                prompts=prompts,
                mask=original_mask,
            )
        
        ema_u = ema_model(
                z,
                t,
                r,
                cache=cache,
                cond=bn,
                spks=spks,
                prompts=prompts,
                mask=original_mask,
        )
        
        pred = z - u
        ema_pred = z - ema_u
        # error = u - stopgrad(u_tgt)
        error = u - v
        

        
        # loss = F.l1_loss(x, pred)
        loss = adaptive_l2_loss(error, original_mask, gamma=1 - self.p)

        mse_val = (stopgrad(error) ** 2)
        mse_val = torch.mean(mse_val, dim=-1)
        mse_val = mse_val[original_mask].mean()

        return loss, mse_val, pred, ema_pred, original_mask
    



    def discrimi(
        self,
        model,
        real,
        generate,
        x:torch.Tensor,
        bn: torch.Tensor,
        spks: torch.Tensor,
        prompts: torch.Tensor,
        inputs_length: torch.tensor,
        ):
        batch_size, seq_len, _ = x.size()
        device = x.device

        t, r = self.sample_t_r(batch_size, device)

        t_ = rearrange(t, "b -> b 1 1")
        r_ = rearrange(r, "b -> b 1 1")

        e = torch.randn_like(x)
        # x = self.normer.norm(x)

        original_mask = lens_to_mask(inputs_length, length=seq_len)
        cache = x
        
        z_r = (1 - t_) * real + t_ * e
        z_g = (1 - t_) * generate + t_ * e

        scores_r = model(
                z_r,
                t,
                r,
                cache=cache,
                cond=bn,
                spks=spks,
                prompts=prompts,
                mask=original_mask,
            )
        
        scores_g = model(
                z_g,
                t,
                r,
                cache=cache,
                cond=bn,
                spks=spks,
                prompts=prompts,
                mask=original_mask,
            )

        sigma = 0.01
        noise = torch.randn_like(z_r) * sigma
        perturbed_z_r = z_r + noise

        scores_r_perturbed = model(
                perturbed_z_r,
                t,
                r,
                cache=cache,
                cond=bn,
                spks=spks,
                prompts=prompts,
                mask=original_mask,
            )


        return scores_r, scores_g, scores_r_perturbed
    

    def discrimi_only_one(
        self,
        model,
        generate,
        x:torch.Tensor,
        bn: torch.Tensor,
        spks: torch.Tensor,
        prompts: torch.Tensor,
        inputs_length: torch.tensor,
        ):
        batch_size, seq_len, _ = x.size()
        device = x.device

        t, r = self.sample_t_r(batch_size, device)

        t_ = rearrange(t, "b -> b 1 1")
        r_ = rearrange(r, "b -> b 1 1")

        e = torch.randn_like(x)
        # x = self.normer.norm(x)

        original_mask = lens_to_mask(inputs_length, length=seq_len)
        cache = x
        
        z_g = (1 - t_) * generate + t_ * e

        
        scores_g = model(
                z_g,
                t,
                r,
                cache=cache,
                cond=bn,
                spks=spks,
                prompts=prompts,
                mask=original_mask,
            )




        return scores_g