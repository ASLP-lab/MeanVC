
import os
import json
import argparse
import glob
from src.model import DiT
from src.model.utils import load_checkpoint
import numpy as np
import torch
from tqdm import tqdm
import torchaudio
from vocos.pretrained import Vocos


def inference(model, vocos, bns, spk_emb, chunk_size, steps, spk_result_dir, device):   
    time_points = torch.linspace(1.0, 0.0, steps + 1, device=device)
            
    for bn_path in tqdm(bns):
        bn = torch.from_numpy(np.load(bn_path)).to(device)
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
            
            x = torch.randn(bn_chunk.shape[1], 80, device=device, dtype=bn_chunk.dtype).unsqueeze(0)
            cfg_mask = torch.ones([x.shape[0]], dtype=torch.bool, device=device)
            
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
        
        mel = x_pred.transpose(1,2)
        mel = (mel + 1) / 2
        y_g_hat = vocos.decode(mel)
        spk_result_wav_dir = spk_result_dir + "_wav"
        os.makedirs(spk_result_wav_dir, exist_ok=True)
        wav_output_path = os.path.join(spk_result_wav_dir, base_filename + ".wav")
        torchaudio.save(wav_output_path, y_g_hat.cpu(), 16000)
            
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-config', type=str, default=None)
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--bn-path', type=str, default=None)
    parser.add_argument('--spk-emb-path', type=str, default=None)
    parser.add_argument('--chunk-size', type=int, default=1)
    parser.add_argument('--steps', type=int, default=1)

    args = parser.parse_args()
    
    output_dir = args.output_dir
    bn_path = args.bn_path
    spk_emb_path = args.spk_emb_path
    chunk_size = args.chunk_size
    steps = args.steps
    
    
    with open(args.model_config) as f:
        model_config = json.load(f)

    model_cls = DiT
    ckpt_path = args.ckpt_path
    
    device='cuda'
    dit_model = model_cls(**model_config["model"])
    total_params = sum(p.numel() for p in dit_model.parameters())
    print(f"Total parameters: {total_params}")
    dit_model = dit_model.to(device)
    dit_model = load_checkpoint(dit_model, ckpt_path, device=device, use_ema=True)
    dit_model = dit_model.float()

    vocos = Vocos.load_selfckpt("/vocoder/vocos-main/logs/vc_10ms/version_0").to(device)

    bns = [path for path in glob.glob(bn_path + "/*.npy")]
    
    spk_emb = np.load(spk_emb_path)
    spk_emb = torch.from_numpy(spk_emb).to(device)
    if len(spk_emb.shape) == 1:
        spk_emb = spk_emb.unsqueeze(0)
    
    inference(model=dit_model,
              vocos=vocos,
              bns=bns,
              spk_emb=spk_emb,
              chunk_size=chunk_size,
              steps=steps,
              spk_result_dir=output_dir,
              device=device
            )
