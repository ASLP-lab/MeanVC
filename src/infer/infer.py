
import os
import json
import argparse
import glob
from src.infer.dit_kvcache import DiT
from src.model.utils import load_checkpoint
import numpy as np
import torch
import time
from tqdm import tqdm
import torchaudio

C_KV_CACHE_MAX_LEN = 100

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

@torch.inference_mode()
def inference(model, vocos, bn_path, spk_emb, prompt_mel, chunk_size, steps, device):
    if steps == 1:
        timesteps = torch.tensor([1.0, 0.0], device=device)
    elif steps == 2:
        timesteps = torch.tensor([1.0, 0.8, 0.0], device=device)
    else:
        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)
    bn = torch.from_numpy(np.load(bn_path)).to(device)
    bn = bn.unsqueeze(0)
    bn = bn.transpose(1, 2)
    bn_interpolate = torch.nn.functional.interpolate(bn, size=int(bn.shape[2] * 4), mode='linear', align_corners=True)
    bn = bn_interpolate.transpose(1, 2)

    seq_len = bn.shape[1]         
    cache = None
    x_pred = []
    B = 1
    offset = 0
    kv_cache = None
    
    s_t_item = time.time()
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        bn_chunk = bn[:, start:end]
        
        
        x = torch.randn(B, bn_chunk.shape[1], 80, device=device, dtype=bn_chunk.dtype)
        
        for i in range(steps):
            t = timesteps[i]
            r = timesteps[i+1]
            t_tensor = torch.full((B,), t, device=x.device)
            r_tensor = torch.full((B,), r, device=x.device)

            u, tmp_kv_cache = model(x, t_tensor, r_tensor, cache=cache, cond=bn_chunk, spks=spk_emb,
                prompts=prompt_mel, offset=offset, is_inference=True, kv_cache=kv_cache)
            x = x - (t - r) * u

        kv_cache = tmp_kv_cache
        offset += x.shape[1]
        cache = x
        x_pred.append(x)
        
        if offset > 40 and kv_cache[0][0].shape[2] > C_KV_CACHE_MAX_LEN:
            for i in range(len(kv_cache)):
                new_k = kv_cache[i][0][:, :, -C_KV_CACHE_MAX_LEN:, :]
                new_v = kv_cache[i][1][:, :, -C_KV_CACHE_MAX_LEN:, :]
                kv_cache[i] = (new_k, new_v)
    x_pred = torch.cat(x_pred, dim=1)
    mel = x_pred.transpose(1,2)
    mel = (mel + 1) / 2
    y_g_hat = vocos.decode(mel)
    time_item = time.time() - s_t_item

    return mel, y_g_hat, time_item
    
def inference_list(model, vocos, bns, spk_emb, prompt_mel, chunk_size, steps, spk_result_dir, device):
  
    rtfs = []
    all_duration = 0
    all_time = 0
    
    for bn_path in tqdm(bns):
        
        mel, wav, time_item = inference(model, vocos, bn_path, spk_emb, prompt_mel, chunk_size, steps, device)
        
        base_filename = os.path.basename(bn_path).split(".")[0]
        mel_output_path = os.path.join(spk_result_dir, base_filename + ".npy")
        np.save(mel_output_path, mel.cpu().numpy())
        
        
        spk_result_wav_dir = spk_result_dir + "_wav"
        os.makedirs(spk_result_wav_dir, exist_ok=True)
        wav_output_path = os.path.join(spk_result_wav_dir, base_filename + ".wav")
        torchaudio.save(wav_output_path, wav.cpu(), 16000)
        duration = wav.shape[1] / 16000
        
        all_duration += duration
        all_time += time_item
        rtf = time_item / duration
        rtfs.append(rtf)
    print("RTF: ", all_time / all_duration)
    print("mean rtf: ", np.mean(rtfs))
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-config', type=str, default=None)
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--vocoder-ckpt-path', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--bn-path', type=str, default=None)
    parser.add_argument('--spk-emb-path', type=str, default=None)
    parser.add_argument('--prompt-path', type=str, default=None)
    parser.add_argument('--chunk-size', type=int, default=1)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
        
    setup_seed(args.seed)
    
    output_dir = args.output_dir
    bn_path = args.bn_path
    spk_emb_path = args.spk_emb_path
    prompt_path = args.prompt_path
    chunk_size = args.chunk_size
    steps = args.steps
    
    
    with open(args.model_config) as f:
        model_config = json.load(f)

    model_cls = DiT
    ckpt_path = args.ckpt_path
    
    device = 'cpu'
    dit_model = model_cls(**model_config["model"])
    total_params = sum(p.numel() for p in dit_model.parameters())
    print(f"Total parameters: {total_params}")
    dit_model = dit_model.to(device)
    dit_model = load_checkpoint(dit_model, ckpt_path, device=device, use_ema=False)
    dit_model = dit_model.float()
    dit_model.eval()
    vocos = torch.jit.load(args.vocoder_ckpt_path).to(device)

    bns = [path for path in glob.glob(bn_path + "/*.npy")]
    
    spk_emb = np.load(spk_emb_path)
    spk_emb = torch.from_numpy(spk_emb).to(device)
    if len(spk_emb.shape) == 1:
        spk_emb = spk_emb.unsqueeze(0)
        
    prompt_mel = np.load(prompt_path)
    prompt_mel = torch.from_numpy(prompt_mel).to(device)
    if len(prompt_mel.shape) < 3:
        prompt_mel = prompt_mel.unsqueeze(0)
    if prompt_mel.shape[1] == 80:
        prompt_mel = prompt_mel.transpose(1, 2)
    
    inference_list(model=dit_model,
              vocos=vocos,
              bns=bns,
              spk_emb=spk_emb,
              prompt_mel=prompt_mel,
              chunk_size=chunk_size,
              steps=steps,
              spk_result_dir=output_dir,
              device=device
            )
