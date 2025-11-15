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
import librosa
import torchaudio.compliance.kaldi as kaldi
import torch.nn as nn
from librosa.filters import mel as librosa_mel_fn
from src.runtime.speaker_verification.verification import init_model as init_sv_model

C_KV_CACHE_MAX_LEN = 100

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True


def _amp_to_db(x, min_level_db):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    min_level = torch.ones_like(x) * min_level
    return 20 * torch.log10(torch.maximum(min_level, x))


def _normalize(S, max_abs_value, min_db):
    return torch.clamp((2 * max_abs_value) * ((S - min_db) / (-min_db)) - max_abs_value, -max_abs_value, max_abs_value)


class MelSpectrogramFeatures(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=1024, win_size=640, hop_length=160, n_mels=80, fmin=0, fmax=8000, center=True):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.mel_basis = {}
        self.hann_window = {}
        

    def forward(self, y):
        dtype_device = str(y.dtype) + '_' + str(y.device)
        fmax_dtype_device = str(self.fmax) + '_' + dtype_device
        wnsize_dtype_device = str(self.win_size) + '_' + dtype_device
        if fmax_dtype_device not in self.mel_basis:
            mel = librosa_mel_fn(sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
            self.mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
        if wnsize_dtype_device not in self.hann_window:
            self.hann_window[wnsize_dtype_device] = torch.hann_window(self.win_size).to(dtype=y.dtype, device=y.device)

        spec = torch.stft(y, self.n_fft, hop_length=self.hop_length, win_length=self.win_size, window=self.hann_window[wnsize_dtype_device],
                        center=self.center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

        spec = torch.matmul(self.mel_basis[fmax_dtype_device], spec)

        spec = _amp_to_db(spec, -115) - 20
        spec = _normalize(spec, 1, -115)
        return spec


def extract_fbanks(wav, sample_rate=16000, mel_bins=80, frame_length=25, frame_shift=12.5):
    wav = wav * (1 << 15)
    wav = torch.from_numpy(wav).unsqueeze(0)
    fbanks = kaldi.fbank(
        wav,
        frame_length=frame_length,
        frame_shift=frame_shift,
        snip_edges=True,
        num_mel_bins=mel_bins,
        energy_floor=0.0,
        dither=0.0,
        sample_frequency=sample_rate,
    )
    fbanks = fbanks.unsqueeze(0)
    return fbanks


def extract_features_from_audio(source_path, reference_path, asr_model, sv_model, mel_extractor, device):

    source_wav, _ = librosa.load(source_path, sr=16000)
    source_fbanks = extract_fbanks(source_wav, frame_shift=10).float().to(device)
    
    with torch.no_grad():
        
        offset = 0
        decoding_chunk_size = 5
        num_decoding_left_chunks = 2
        subsampling = 4
        context = 7  # Add current frame
        stride = subsampling * decoding_chunk_size
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        att_cache = torch.zeros((0, 0, 0, 0), device=device)
        cnn_cache = torch.zeros((0, 0, 0, 0), device=device)
        
        bn_chunks = []
        
        for i in range(0, source_fbanks.shape[1], stride):
            fbank_chunk = source_fbanks[:, i:i+decoding_window, :]
            if fbank_chunk.shape[1] < required_cache_size:
                break
                
            encoder_output, att_cache, cnn_cache = asr_model.forward_encoder_chunk(
                fbank_chunk, offset, required_cache_size, att_cache, cnn_cache
            )
            offset += encoder_output.size(1)
            bn_chunks.append(encoder_output)
        
        bn = torch.cat(bn_chunks, dim=1)  # [1, T, 256]

        bn = bn.transpose(1, 2)
        bn = torch.nn.functional.interpolate(bn, size=int(bn.shape[2] * 4), mode='linear', align_corners=True)
        bn = bn.transpose(1, 2)
    

    ref_wav, _ = librosa.load(reference_path, sr=16000)
    ref_wav_tensor = torch.from_numpy(ref_wav).unsqueeze(0).to(device)
    
    with torch.no_grad():

        spk_emb = sv_model(ref_wav_tensor)  # [1, 256]
        
        prompt_mel = mel_extractor(ref_wav_tensor)  # [1, 80, T]
        prompt_mel = prompt_mel.transpose(1, 2)  # [1, T, 80]
    
    return bn, spk_emb, prompt_mel


@torch.inference_mode()
def inference(model, vocos, bn, spk_emb, prompt_mel, chunk_size, steps, device):
    if steps == 1:
        timesteps = torch.tensor([1.0, 0.0], device=device)
    elif steps == 2:
        timesteps = torch.tensor([1.0, 0.8, 0.0], device=device)
    else:
        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)

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
        
        if offset > 40 and kv_cache is not None and kv_cache[0][0].shape[2] > C_KV_CACHE_MAX_LEN:
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

    
def inference_list(model, vocos, asr_model, sv_model, mel_extractor, sources, reference_path, chunk_size, steps, output_dir, device):

    rtfs = []
    all_duration = 0
    all_time = 0
    
    ref_wav, _ = librosa.load(reference_path, sr=16000)
    ref_wav_tensor = torch.from_numpy(ref_wav).unsqueeze(0).to(device)
    

    for source_path in tqdm(sources):
        print(f"\nProcessing: {source_path}")
        
        bn, spk_emb, prompt_mel = extract_features_from_audio(
            source_path, reference_path, asr_model, sv_model, mel_extractor, device
        )
        
        mel, wav, time_item = inference(model, vocos, bn, spk_emb, prompt_mel, chunk_size, steps, device)
        
        base_filename = os.path.basename(source_path).split(".")[0]
        
        mel_output_path = os.path.join(output_dir, base_filename + ".npy")
        np.save(mel_output_path, mel.cpu().numpy())
        
        wav_output_dir = output_dir + "_wav"
        os.makedirs(wav_output_dir, exist_ok=True)
        wav_output_path = os.path.join(wav_output_dir, base_filename + ".wav")
        torchaudio.save(wav_output_path, wav.cpu(), 16000)
        
        duration = wav.shape[1] / 16000
        all_duration += duration
        all_time += time_item
        rtf = time_item / duration
        rtfs.append(rtf)
        
        print(f"Duration: {duration:.2f}s, Time: {time_item:.2f}s, RTF: {rtf:.4f}")
    
    print(f"\n=== Results ===")
    print(f"Total RTF: {all_time / all_duration:.4f}")
    print(f"Mean RTF: {np.mean(rtfs):.4f}")

            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-config', type=str, default=None)
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--asr-ckpt-path', type=str, default='src/ckpt/fastu2++.pt')
    parser.add_argument('--sv-ckpt-path', type=str, default='src/runtime/speaker_verification/ckpt/wavlm_large_finetune.pth')
    parser.add_argument('--vocoder-ckpt-path', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--source-path', type=str, default=None, help='Source audio file or directory')
    parser.add_argument('--reference-path', type=str, required=True, help='Reference audio file')
    parser.add_argument('--chunk-size', type=int, default=20)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
        
    setup_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = 'cpu'

    with open(args.model_config) as f:
        model_config = json.load(f)

    model_cls = DiT
    dit_model = model_cls(**model_config["model"])
    total_params = sum(p.numel() for p in dit_model.parameters())
    print(f"Total parameters: {total_params}")
    dit_model = dit_model.to(device)
    dit_model = load_checkpoint(dit_model, args.ckpt_path, device=device, use_ema=False)
    dit_model = dit_model.float()
    dit_model.eval()

    vocos = torch.jit.load(args.vocoder_ckpt_path).to(device)

    asr_model = torch.jit.load(args.asr_ckpt_path).to(device)

    sv_model = init_sv_model('wavlm_large', args.sv_ckpt_path)
    sv_model = sv_model.to(device)
    sv_model.eval()
    
    mel_extractor = MelSpectrogramFeatures(
        sample_rate=16000, n_fft=1024, win_size=640, hop_length=160, 
        n_mels=80, fmin=0, fmax=8000, center=True
    ).to(device)

    if os.path.isdir(args.source_path):
        sources = glob.glob(os.path.join(args.source_path, "*.wav"))
    else:
        sources = [args.source_path]
    
    print(f"Found {len(sources)} source audio files")

    inference_list(
        model=dit_model,
        vocos=vocos,
        asr_model=asr_model,
        sv_model=sv_model,
        mel_extractor=mel_extractor,
        sources=sources,
        reference_path=args.reference_path,
        chunk_size=args.chunk_size,
        steps=args.steps,
        output_dir=args.output_dir,
        device=device
    )