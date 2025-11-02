
import sys
import numpy as np
from tqdm import tqdm
import torch
import numpy as np
import os
import librosa
import torchaudio.compliance.kaldi as kaldi
from functools import partial
import argparse

from multiprocessing import Pool
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


def extract_fbanks(
    wav, sample_rate=16000, mel_bins=80, frame_length=25, frame_shift=12.5
):
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


asr = torch.jit.load('src/runtime/ckpt/fastu2++.pt').to(device)

def extract_bn(wav_path):
    
    in_wav, _ = librosa.load(wav_path, sr=16000)

    fbanks = extract_fbanks(in_wav, frame_shift=10).float().to(device)
    offset = 0
    decoding_chunk_size = 4
    num_decoding_left_chunks = 2
    subsampling = 4
    context = 7  # Add current frame
    stride = subsampling * decoding_chunk_size
    required_cache_size = decoding_chunk_size * num_decoding_left_chunks
    decoding_window = (decoding_chunk_size - 1) * subsampling + context
    att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device='cpu')
    cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device='cpu')
    bns = []
    for i in range(0, fbanks.shape[1], 16):
        fbank = fbanks[:, i:i+19, :]
        if fbank.shape[1] < 10:
            break
        (encoder_output, att_cache, cnn_cache) = asr.forward_encoder_chunk(
            fbank, offset, required_cache_size, att_cache, cnn_cache)

        bns.append(encoder_output)

        offset += encoder_output.size(1)
    
    bn = torch.cat(bns, dim=1)
    bn = bn.squeeze()
    bn = bn.detach().cpu().numpy()
    return bn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="wav dir")
    parser.add_argument("--output_dir", type=str, help="out bn feature dir")
    parser.add_argument("--filelist", type=str, help="filelst", default=None)


    args = parser.parse_args()

    data_root = args.input_dir
    out_data_root = args.output_dir
    filelist = args.filelist
    
    os.makedirs(out_data_root, exist_ok=True)

    if filelist is not None:
        filelist = open(filelist).read().splitlines()
        generator = filelist
    else:
        generator = os.listdir(data_root)

    for wav_filename in tqdm(generator):
        wav_path = os.path.join(data_root, wav_filename)
        try:
            bn = extract_bn(wav_path)
        except:
            print("!!!!!!!", wav_path)
        np.save(os.path.join(out_data_root, wav_filename.split('.')[0]), bn)