import torch
import torch.nn as nn
import numpy as np
import librosa
from librosa.filters import mel as librosa_mel_fn
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import soundfile as sf
import traceback


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


class MelExtractor:
    
    def __init__(self, 
                 sample_rate=16000,
                 n_fft=1024,
                 win_size=640,
                 hop_length=160,
                 n_mels=80,
                 fmin=0,
                 fmax=8000,
                 device='cuda'):

        self.device = device
        self.sample_rate = sample_rate
        self.mel_extractor = MelSpectrogramFeatures(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_size=win_size,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            center=True
        ).to(device)
        
    def load_audio(self, audio_path):
        try:
            audio, sr = sf.read(audio_path)

            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
                
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
            return audio
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
    
    def extract_mel(self, audio):
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mel = self.mel_extractor(audio_tensor)

        mel = mel.squeeze(0).cpu().numpy().T
        
        return mel
    
    def process_file(self, audio_path, output_path):
        try:

            audio = self.load_audio(audio_path)
            if audio is None:
                return False

            mel = self.extract_mel(audio)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, mel)
            
            return True
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            traceback.print_exc()
            return False
    
    def process_directory(self, input_dir, output_dir, audio_extensions=['.wav', '.flac', '.mp3', '.m4a'], recursive=True):

        input_path = Path(input_dir)
        output_path = Path(output_dir)

        audio_files = []
        if recursive:
            for ext in audio_extensions:
                audio_files.extend(list(input_path.rglob(f'*{ext}')))
        else:
            for ext in audio_extensions:
                audio_files.extend(list(input_path.glob(f'*{ext}')))
        
        print(f"Found {len(audio_files)} audio files")

        success_count = 0
        fail_count = 0
        
        for audio_file in tqdm(audio_files, desc="Extracting Mel features"):
            relative_path = audio_file.relative_to(input_path)
            output_file = output_path / relative_path.with_suffix('.npy')

            if output_file.exists():
                success_count += 1
                continue

            if self.process_file(str(audio_file), str(output_file)):
                success_count += 1
            else:
                fail_count += 1
        
        print(f"\nProcessing completed!")
        print(f"Success: {success_count}/{len(audio_files)}")
        print(f"Failed: {fail_count}/{len(audio_files)}")
        
        return success_count, fail_count
    
    def process_file_list(self, file_list_path, input_dir, output_dir):

        input_path = Path(input_dir)
        output_path = Path(output_dir)

        with open(file_list_path, 'r', encoding='utf-8') as f:
            file_list = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(file_list)} files in list")
        
        success_count = 0
        fail_count = 0
        
        for relative_path in tqdm(file_list, desc="Extracting Mel features"):
            audio_file = input_path / relative_path
            output_file = output_path / Path(relative_path).with_suffix('.npy')

            if output_file.exists():
                success_count += 1
                continue

            if not audio_file.exists():
                print(f"File not found: {audio_file}")
                fail_count += 1
                continue

            if self.process_file(str(audio_file), str(output_file)):
                success_count += 1
            else:
                fail_count += 1
        
        print(f"\nProcessing completed!")
        print(f"Success: {success_count}/{len(file_list)}")
        print(f"Failed: {fail_count}/{len(file_list)}")
        
        return success_count, fail_count


def main():
    parser = argparse.ArgumentParser(description='Extract Mel-spectrogram features from audio files')
    parser.add_argument('--input_dir', type=str, required=True, help='Input audio directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output mel feature directory')
    parser.add_argument('--file_list', type=str, default=None, help='Optional file list (one relative path per line)')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Target sample rate')
    parser.add_argument('--n_fft', type=int, default=1024, help='FFT size')
    parser.add_argument('--win_size', type=int, default=640, help='Window size')
    parser.add_argument('--hop_length', type=int, default=160, help='Hop length (10ms at 16kHz)')
    parser.add_argument('--n_mels', type=int, default=80, help='Number of mel filters')
    parser.add_argument('--fmin', type=int, default=0, help='Minimum frequency')
    parser.add_argument('--fmax', type=int, default=8000, help='Maximum frequency')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--recursive', action='store_true', help='Process subdirectories recursively')
    parser.add_argument('--extensions', type=str, default='.wav,.flac,.mp3,.m4a', help='Audio file extensions (comma-separated)')
    
    args = parser.parse_args()

    extractor = MelExtractor(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        win_size=args.win_size,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        device=args.device
    )

    if args.file_list:
        extractor.process_file_list(args.file_list, args.input_dir, args.output_dir)
    else:
        extensions = [ext.strip() for ext in args.extensions.split(',')]
        extractor.process_directory(
            args.input_dir, 
            args.output_dir, 
            audio_extensions=extensions,
            recursive=args.recursive
        )


if __name__ == "__main__":
  
    main()