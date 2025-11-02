from __future__ import annotations

import os
import gc
from tqdm import tqdm
import wandb

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR

import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from src.dataset.dataset import DiffusionDataset
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from ema_pytorch import EMA
from src.model.dit_discriminator import DiT_dis
from src.model.loss import GANLoss
from src.model import MeanFlow
from src.model.utils import exists, default, plot_spectrogram, optimized_scale
from src.eval.verification import init_model
from src.eval.run_wer import process_one, load_zh_model
import zhconv
from vocos.pretrained import Vocos
import torchaudio
import time

# trainer


class Trainer:
    def __init__(
        self,
        model,
        args,
        model_config,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        checkpoint_path=None,
        batch_size=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        wandb_project="meanvc",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        last_per_steps=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        reset_lr: bool = False,
        grad_ckpt: bool = False
    ):
        self.args = args
        self.expname = wandb_run_name

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False, )

        logger = "wandb" if wandb.api.api_key else None
        #logger = None
        print(f"Using logger: {logger}")
        self.accelerator = Accelerator(
            log_with=logger,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        if logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config={
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size": batch_size,
                    "max_samples": self.args.max_len,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "gpus": self.accelerator.num_processes,
                },
            )

        self.precision = self.accelerator.state.mixed_precision
        self.precision = self.precision.replace("no", "fp32")
        print("!!!!!!!!!!!!!!!!!", self.precision)

        self.model = model
        
        self.args.flow_ratio
        
        self.meanflow = MeanFlow(
                        flow_ratio=self.args.flow_ratio,
                        time_dist=['lognorm', -0.4, 1.0],
                        cfg_ratio=self.args.cfg_ratio,
                        cfg_scale=self.args.cfg_scale,
                        cfg_uncond='u',
                        p=self.args.p,)
        
        self.discriminator = DiT_dis(**model_config["model"])
        
        self.gan_loss = GANLoss()
        
        #self.model = torch.compile(model)

        #self.dataloader = dataloader

        # if self.is_main:
        self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)

           # self.ema_model.to(self.accelerator.device)
        if self.accelerator.state.distributed_type in ["DEEPSPEED", "FSDP"]:
            self.ema_model.half()

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.last_per_steps = default(last_per_steps, save_per_updates * grad_accumulation_steps)
        self.checkpoint_path = default(checkpoint_path, "ckpts/meanvc")

        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.reset_lr = reset_lr
        
        self.grad_ckpt = grad_ckpt

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
            self.dis_optimizer = AdamW(self.discriminator.parameters(), lr=learning_rate)
        #self.optimizer = FusedAdam(model.parameters(), lr=learning_rate)

        #self.model = torch.compile(self.model)
        if self.accelerator.state.distributed_type == "DEEPSPEED":
            self.accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = batch_size
        
        self.get_dataloader()
        self.get_scheduler()
        self.get_dis_scheduler()
        self.start_step = self.load_checkpoint()
        # self.get_constant_scheduler()

        self.model,self.ema_model, self.optimizer, self.scheduler, self.discriminator, self.dis_optimizer, self.dis_scheduler, self.train_dataloader = self.accelerator.prepare(self.model, self.ema_model, self.optimizer, self.scheduler, self.discriminator, self.dis_optimizer, self.dis_scheduler, self.train_dataloader)

    def get_scheduler(self):
        warmup_steps = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        total_steps = len(self.train_dataloader) * self.epochs / self.grad_accumulation_steps
        total_steps = 1000000 * self.accelerator.num_processes
        decay_steps = total_steps - warmup_steps
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.2, total_iters=decay_steps)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps]
        )
        
    def get_dis_scheduler(self):
        warmup_steps = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        total_steps = len(self.train_dataloader) * self.epochs / self.grad_accumulation_steps
        total_steps = 1000000 * self.accelerator.num_processes
        decay_steps = total_steps - warmup_steps
        warmup_scheduler = LinearLR(self.dis_optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps)
        decay_scheduler = LinearLR(self.dis_optimizer, start_factor=1.0, end_factor=0.2, total_iters=decay_steps)
        # constant_scheduler = ConstantLR(self.optimizer, factor=1, total_iters=decay_steps)
        self.dis_scheduler = SequentialLR(
            self.dis_optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps]
        )

    def get_constant_scheduler(self):
        total_steps = len(self.train_dataloader) * self.epochs / self.grad_accumulation_steps
        self.scheduler = ConstantLR(self.optimizer, factor=1, total_iters=total_steps)

    def get_dataloader(self):
        dd = DiffusionDataset(*DiffusionDataset.init_data(self.args.dataset_path), \
                                feature_list=self.args.feature_list, \
                                additional_feature_list=self.args.additional_feature_list, \
                                feature_pad_values=self.args.feature_pad_values, \
                                max_len=self.args.max_len)
        
        # start_time = time.time()
        self.train_dataloader = DataLoader(
            dataset=dd,
            batch_size=self.args.batch_size,    
            shuffle=True,
            num_workers=self.args.num_workers,     
            pin_memory=True,
            collate_fn=dd.custom_collate_fn,
            persistent_workers=self.args.num_workers > 0,
        )


    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, step, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                discriminator_state_dict=self.accelerator.unwrap_model(self.discriminator).state_dict(),
                dis_optimizer_state_dict=self.dis_optimizer.state_dict(),
                dis_scheduler_state_dict=self.dis_scheduler.state_dict(),
                step=step,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at step {step}")
            else:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{step}.pt")

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not os.listdir(self.checkpoint_path)
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            latest_checkpoint = sorted(
                [f for f in os.listdir(self.checkpoint_path) if f.endswith(".pt")],
                key=lambda x: int("".join(filter(str.isdigit, x))),
            )[-1]
        # checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location=self.accelerator.device)  # rather use accelerator.load_state ಥ_ಥ
        checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location="cpu")

        if self.is_main:
            #del(checkpoint["ema_model_state_dict"]["ema_model.transformer.text_embed.text_embed.weight"])
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"], strict=False)

        # if "step" in checkpoint:
        if not self.reset_lr:
            #del(checkpoint["model_state_dict"]["transformer.text_embed.text_embed.weight"])
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # if self.scheduler and not self.reset_lr:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.accelerator.unwrap_model(self.discriminator).load_state_dict(checkpoint["discriminator_state_dict"], strict=False)
            self.dis_optimizer.load_state_dict(checkpoint["dis_optimizer_state_dict"])
            self.dis_scheduler.load_state_dict(checkpoint["dis_scheduler_state_dict"])
            step = checkpoint["step"]
        else:
            # checkpoint["model_state_dict"] = {
            #     k.replace("ema_model.", ""): v
            #     for k, v in checkpoint["ema_model_state_dict"].items()
            #     if k not in ["initted", "step"]
            # }
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"], strict=False)
            step = 0

        del checkpoint
        gc.collect()
        print("step", step)
        return step

    def validate(self, step):
        self.model.eval()
        
        model = self.accelerator.unwrap_model(self.model)
        vocos = Vocos.load_selfckpt("/vc_10ms/version_0").to(self.model.device)
        spks_file = "/val/spks.lst"
        bn_file = "/val/bn/bn_40ms_160ms.scp"
        text_path = "/val/text.txt"
        prompt_dir = "/val/mels_10ms"
        
        # smos
        sv_model = init_model('wavlm_large', '/test/smos/ckpt/wavlm_large_finetune.pth')
        sv_model.eval()
        sv_model.to(self.model.device)
        ssmi_dict = {}
        
        # cer
        asr_model = load_zh_model(str(self.model.device))
        wer_dict = {}
        utt2txt = {}
        text_all = ''
        for line in open(text_path, 'r').readlines():
            utt, txt = line.strip().split('|')
            utt2txt[utt] = txt
            text_all += utt2txt[utt]
            
        spks = [line.strip() for line in open(spks_file, "r")]
        bns = [line.strip() for line in open(bn_file, "r")]
        
        steps = self.args.steps
        chunk_size = self.args.chunk_size
        cfg_strength = self.args.cfg_strength
        time_points = torch.linspace(1.0, 0.0, steps + 1, device=self.model.device)
         
        with torch.no_grad():
            for i, spk in enumerate(spks):
                ssmi_list = []
                transcription_all = ""
                spk_filename = os.path.basename(spk).split(".")[0]
                
                # load prompt mel
                prompt_path = os.path.join(prompt_dir, spk_filename + ".npy")
                prompt_mel = np.load(prompt_path)
                prompt_mel = torch.from_numpy(prompt_mel).to(self.model.device)
                prompt_mel = prompt_mel.unsqueeze(0)
                if prompt_mel.shape[1] == 80:
                    prompt_mel = prompt_mel.transpose(1, 2)
                
                # load speaker embedding
                spk_result_dir = os.path.join(self.args.result_dir, self.expname, str(step), spk_filename)
                os.makedirs(spk_result_dir, exist_ok=True)
                spk_emb = np.load(spk)
                spk_emb = torch.from_numpy(spk_emb).to(self.model.device)
                if len(spk_emb.shape) == 1:
                    spk_emb = spk_emb.unsqueeze(0)
                
                for bn_path in bns:
                    bn = torch.from_numpy(np.load(bn_path)).to(self.model.device)
                    bn = bn.unsqueeze(0)
                    bn = bn.transpose(1, 2)
                    bn_interpolate = torch.nn.functional.interpolate(bn, size=int(bn.shape[2] * 4), mode='linear', align_corners=True)
                    bn = bn_interpolate.transpose(1, 2)
                    
                    seq_len = bn.shape[1]
                    num_chunks = seq_len // chunk_size
                    if seq_len % chunk_size != 0:
                        num_chunks += 1
                        
                    cache = None
                    x_pred_collect = []
                    
                    offset = 0
                    for chunk_id in range(num_chunks):
                        start = chunk_id * chunk_size
                        end = min(start + chunk_size, seq_len)
                        bn_chunk = bn[:, start:end]
                        if chunk_id == 0:
                            cache = None
                        
                        x = torch.randn(bn_chunk.shape[1], 80, device=self.model.device, dtype=bn_chunk.dtype).unsqueeze(0)
                        cfg_mask = torch.ones([x.shape[0]], dtype=torch.bool, device=self.model.device)
                        
                        for i in range(steps):
                            t = time_points[i]
                            r = time_points[i+1]
                            t_tensor = torch.full((x.size(0),), t, device=x.device)
                            r_tensor = torch.full((x.size(0),), r, device=x.device)
                            with torch.inference_mode():
                                u = model(
                                    x,
                                    t_tensor,
                                    r_tensor,
                                    cache=cache,
                                    cond=bn_chunk,
                                    spks=spk_emb,
                                    prompts=prompt_mel,
                                    offset=offset,
                                    is_inference=True,
                                )
                                
                                x = x - (t - r) * u
                            
                        offset += x.shape[1]
                        if cache == None:
                            cache = x
                        else:
                            cache = torch.cat([cache, x], dim=1)
                            
                        x_pred_collect.append(x) 
                    
                    x_pred = torch.cat(x_pred_collect, dim=1)
                    
                    base_filename = os.path.basename(bn_path).split(".")[0]
                    mel_output_path = os.path.join(spk_result_dir, base_filename + ".npy")
                    np.save(mel_output_path, x_pred.cpu().numpy())
                    
                    # self.accelerator.log({f"mel_spectrogram_{spk_filename}_{base_filename}": wandb.Image(plot_spectrogram(x_pred.transpose(1,2).squeeze(0).cpu().numpy()))}, step=step)
                    
                    mel = x_pred.transpose(1,2)
                    mel = (mel + 1) / 2
                    y_g_hat = vocos.decode(mel)
                    wav_output_path = os.path.join(spk_result_dir, base_filename + ".wav")
                    torchaudio.save(wav_output_path, y_g_hat.cpu(), 16000)
                    # self.accelerator.log({f"audio_{spk_filename}_{base_filename}": wandb.Audio(wav_output_path, sample_rate=16000)}, step=step)
                    
                    # smi
                    with torch.no_grad():
                        emb = sv_model(y_g_hat)
                    ssmi = F.cosine_similarity(spk_emb, emb)
                    ssmi_list.append(ssmi.item()) 
                    
                    # cer
                    try:
                        res = asr_model.generate(input=wav_output_path, batch_size_s=300)
                        transcription = res[0]["text"]
                        transcription = zhconv.convert(transcription, 'zh-cn')
                    except:
                        print(f"Error processing {wav_output_path}, skipping...")
                        transcription = ""
                        
                    transcription_all += transcription
                
                wer = process_one(transcription_all, text_all)
                wer_dict[spk_filename] = wer
                
                ssmi_mean = np.mean(ssmi_list)
                ssmi_dict[spk_filename] = ssmi_mean
            
        for spk_filename in ssmi_dict:
            ssmi_mean = ssmi_dict[spk_filename]
            wer = wer_dict[spk_filename]
            self.accelerator.log({f"ssmi_{spk_filename}": ssmi_mean, f"wer_{spk_filename}": wer}, step=step)
        self.model.train()
        del sv_model
        del asr_model
        torch.cuda.empty_cache()       
                    
    def train(self, resumable_with_seed: int = None):
        train_dataloader = self.train_dataloader

        start_step = self.start_step
        global_step = start_step
        
        self.model.train()
        
        if resumable_with_seed > 0:
            orig_epoch_step = len(train_dataloader)
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0
        
        print(self.model.device)

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if resumable_with_seed > 0 and epoch == skipped_epoch:
                progress_bar = tqdm(
                    skipped_dataloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                    initial=skipped_batch,
                    total=orig_epoch_step,
                    smoothing=0.15
                )
            else:
                progress_bar = tqdm(
                    train_dataloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                    smoothing=0.15
                )
            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    features = {}
                    for feature_name in self.args.feature_list + self.args.additional_feature_list:
                        features[feature_name] = batch[feature_name]
                    
                    
                    diff_loss, mse_val, pred, ema_pred, original_mask = self.meanflow.loss_ema_one_step_only(self.model, self.ema_model,
                        x=features["mel"],
                        bn=features["bn"],
                        spks=features["xvector"],
                        prompts=features["prompt"],
                        inputs_length=features["inputs_length"])

                    p, p_, p_perturbed = self.meanflow.discrimi(self.discriminator, 
                                                   real = features["mel"], 
                                                   generate = pred.detach(),
                                                   x=features["mel"],
                                                    bn=features["bn"],
                                                    spks=features["xvector"],
                                                    prompts=features["prompt"],
                                                    inputs_length=features["inputs_length"])


                    real_loss, fake_loss = self.gan_loss.disc_loss(p, p_)

                    r1_loss = self.gan_loss.distill_loss(p, p_perturbed)
                    gamma_r1 = 100

                    d_loss = real_loss + fake_loss + gamma_r1 * r1_loss

                    self.accelerator.backward(d_loss)
                    dis_grad_norm = None
                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        dis_grad_norm = self.accelerator.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
                    
                    self.dis_optimizer.step()
                    self.dis_scheduler.step()
                    self.dis_optimizer.zero_grad()

                    p_ = self.meanflow.discrimi_only_one(self.discriminator, 
                                                   generate = pred,
                                                   x=features["mel"],
                                                    bn=features["bn"],
                                                    spks=features["xvector"],
                                                    prompts=features["prompt"],
                                                    inputs_length=features["inputs_length"])
                    
                    adv_loss = 0.5 * self.gan_loss.gen_loss(p_)

                    distill_loss = self.gan_loss.distill_loss(pred, ema_pred)


                    total_loss = diff_loss + adv_loss + distill_loss * 0.05
                    
                    self.accelerator.backward(total_loss)
                    grad_norm = None

                    
                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.is_main:
                    self.ema_model.update()

                global_step += 1

                if self.accelerator.is_local_main_process:
                    log_data = {
                        "loss": diff_loss.item(),
                        "mse_val": mse_val.item(),
                        "r1_loss":r1_loss.item()*100,
                        "real_loss": real_loss.item(),
                        "fake_loss": fake_loss.item(),
                        "dis_loss": d_loss.item(),
                        "adv_loss": adv_loss.item(),
                        "lr": self.scheduler.get_last_lr()[0],
                    }
                    if grad_norm is not None:
                        log_data["grad_norm"] = round(grad_norm.item(), 2)
                    if dis_grad_norm is not None:
                        log_data["dis_grad_norm"] = round(dis_grad_norm.item(), 2)
                        
                    self.accelerator.log(log_data, step=global_step)

                progress_bar.set_postfix(step=str(global_step), loss=diff_loss.item(), mse_val=mse_val.item())

                if global_step % (self.save_per_updates * self.grad_accumulation_steps) == 0:
                    self.save_checkpoint(global_step)
                    try:
                        self.validate(global_step)
                    except Exception as e:
                        print(f"ERROR: {e}")

                        
                    

                if global_step % self.last_per_steps == 0:
                    self.save_checkpoint(global_step, last=True)
                    

        self.save_checkpoint(global_step, last=True)

        self.accelerator.end_training()
