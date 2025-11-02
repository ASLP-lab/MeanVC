import soundfile as sf
import torch
import os
from collections import defaultdict 
import argparse
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torchaudio.transforms import Resample
from .ecapa_tdnn import ECAPA_TDNN_SMALL
import glob

MODEL_LIST = ['ecapa_tdnn', 'hubert_large', 'wav2vec2_xlsr', 'unispeech_sat', "wavlm_base_plus", "wavlm_large"]


def init_model(model_name, checkpoint=None):
    if model_name == 'unispeech_sat':
        config_path = 'config/unispeech_sat.th'
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='unispeech_sat', config_path=config_path)
    elif model_name == 'wavlm_base_plus':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=768, feat_type='wavlm_base_plus', config_path=config_path)
    elif model_name == 'wavlm_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=config_path)
    elif model_name == 'hubert_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='hubert_large_ll60k', config_path=config_path)
    elif model_name == 'wav2vec2_xlsr':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wav2vec2_xlsr', config_path=config_path)
    else:
        model = ECAPA_TDNN_SMALL(feat_dim=40, feat_type='fbank')

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model'], strict=False)
    return model


def get_emb(model, wav, device='cpu', sample_rate=16000):
    wav, sr = sf.read(wav)
    if wav.ndim == 2:
        wav = np.mean(wav, axis=1)

    wav = torch.from_numpy(wav).unsqueeze(0).float().to(device)

    if sr != sample_rate:
        resample = Resample(orig_freq=sr, new_freq=sample_rate).to(device)
        wav = resample(wav)

    with torch.no_grad():
        emb = model(wav)

    return emb


def verification_v1(model, target_spk, wavs, device='cpu', sample_rate=16000):
    target_spk_emb = get_emb(model, target_spk, device, sample_rate)
    spk_sims = {}
    for wav in tqdm(wavs):
        emb = get_emb(model, wav, device, sample_rate)
        spk_sims[wav] = F.cosine_similarity(target_spk_emb, emb)
    return spk_sims


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wavs')
    parser.add_argument('--target_spk')
    parser.add_argument('--output')
    parser.add_argument('--device', default="cuda:0")
    args = parser.parse_args()
    
    wav_dir = args.wavs
    target_spk = args.target_spk
    output = args.output
    device = args.device

    wavs = glob.glob(os.path.join(wav_dir, '*.wav'))

    model = init_model('wavlm_large', '/ckpt/wavlm_large_finetune.pth')
    model.eval()
    model.to(device)

    sims = verification_v1(model, target_spk, wavs, device, 16000)
    # open(output, 'w').writelines([f"{sum(sims.values())/len(list(sims.keys()))}\n", str(sims)+'\n'])
    with open(output, 'w') as f:
        f.write(f"{sum(sims.values())/len(list(sims.keys()))}\n")
        for key, value in sims.items():
            f.write(f"{key}: {value}\n")