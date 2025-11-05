import time
import argparse
import numpy as np
import pyaudio
import torch.nn as nn
from tqdm import tqdm
import torch
from threading import Thread, Lock
import torchaudio.compliance.kaldi as kaldi
import librosa
from librosa.filters import mel as librosa_mel_fn
from speaker_verification.verification import init_model as init_sv_model
import json


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

class VCRunner():
    def __init__(self, target_path, steps=2):
        self.mutex = Lock()
        torch.set_num_threads(1)
        self.sv_model = init_sv_model('wavlm_large', 'src/runtime/speaker_verification/ckpt/wavlm_large_finetune.pth')
        self.sv_model.eval()
        
        
        self.steps = steps
        if self.steps == 1:
            self.timesteps = torch.tensor([1.0, 0.0])
        elif self.steps == 2:
            self.timesteps = torch.tensor([1.0, 0.8, 0.0])
        else:
            self.timesteps = torch.linspace(1.0, 0.0, self.steps + 1)
        
        self.mel_extract = MelSpectrogramFeatures(sample_rate=16000, n_fft=1024, win_size=640, hop_length=160, n_mels=80, fmin=0, fmax=8000, center=True)
        
        self.asr = torch.jit.load('src/ckpt/fastu2++.pt')
        self.vc = torch.jit.load("src/ckpt/meanvc_200ms.pt")
        self.vocoder = torch.jit.load('src/ckpt/vocos.pt')
        

        decoding_chunk_size = 5
        num_decoding_left_chunks = 2
        subsampling = 4
        context = 7  # Add current frame
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        self.required_cache_size = decoding_chunk_size * num_decoding_left_chunks
        self.CHUNK = 160 * stride
        self.vc_chunk = int(decoding_chunk_size * 4)
        self.vocoder_overlap = 3
        upsample_factor = 160
        self.vocoder_wav_overlap = (self.vocoder_overlap - 1) * upsample_factor
        self.down_linspace = torch.linspace(1, 0, steps=self.vocoder_wav_overlap, out=None).numpy()
        self.up_linspace = torch.linspace(0, 1, steps=self.vocoder_wav_overlap, out=None).numpy()

        self.wav_path = target_path
        wav, _ = librosa.load(self.wav_path, sr=16000)
        wav = torch.from_numpy(wav).unsqueeze(0)
        
        spk_emb = self.sv_model(wav)
        self.vc_spk_emb = spk_emb
        
        prompt_mel = self.mel_extract(wav)
        prompt_mel = prompt_mel.transpose(1, 2)
        
        self.vc_prompt_mel = prompt_mel


    def playaudio(self, out_stream, data):
        with self.mutex:
            out_stream.write(data)

    def init_cache(self):
        self.samples_cache_len = 720   # 400 + 2 * 160
        self.samples_cache = None
        

        self.att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device='cpu')
        self.cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device='cpu')
        self.asr_offset = 0
        self.encoder_output_cache = None

        self.vc_offset = 0
        self.vc_cache = None
        self.vc_kv_cache = None

        self.vocoder_cache = None
        self.last_wav = None

        self.need_extra_data = True
    
    def reset_cache(self):
        self.asr_offset = 20
        self.vc_offset = 120



    def inference_one_chunk(self, samples):
        with torch.no_grad():
            if self.samples_cache is None:
                samples = samples
            else: 
                samples = np.concatenate((self.samples_cache, samples))
            self.samples_cache = samples[-self.samples_cache_len:]
            fbanks = extract_fbanks(samples, frame_shift=10).float()   # 23 frame  torch.Size([1, 23, 80])
            (encoder_output, self.att_cache, self.cnn_cache) = self.asr.forward_encoder_chunk(
                fbanks, self.asr_offset, self.required_cache_size, self.att_cache, self.cnn_cache)

            self.asr_offset += encoder_output.size(1)    # 5 frame
            if self.encoder_output_cache is None:
                encoder_output = torch.cat([encoder_output[:, 0:1, :], encoder_output], dim=1)
            else:
                encoder_output = torch.cat([self.encoder_output_cache, encoder_output], dim=1)
            self.encoder_output_cache = encoder_output[:, -1:, :]
            encoder_output_upsample = encoder_output.transpose(1, 2)
            encoder_output_upsample = torch.nn.functional.interpolate(encoder_output_upsample, size=self.vc_chunk + 1, mode='linear', align_corners=True) # 6 -> 21
            encoder_output_upsample = encoder_output_upsample.transpose(1, 2)
            encoder_output_upsample = encoder_output_upsample[:, 1:, :]
            
            
            x = torch.randn(1, encoder_output_upsample.shape[1], 80, device='cpu', dtype=encoder_output_upsample.dtype)
            
            for i in range(self.steps):
                t = self.timesteps[i]
                r = self.timesteps[i+1]
                t_tensor = torch.full((1,), t, device=x.device)
                r_tensor = torch.full((1,), r, device=x.device)
            
                u, tmp_kv_cache = self.vc(x, t_tensor, r_tensor, cache=self.vc_cache, cond=encoder_output_upsample, spks=self.vc_spk_emb,
                    prompts=self.vc_prompt_mel, offset=self.vc_offset, kv_cache=self.vc_kv_cache)
                
                x = x - (t - r) * u
            self.vc_kv_cache = tmp_kv_cache
            self.vc_offset += x.shape[1]
            self.vc_cache = x

            VC_KV_CACHE_MAX_LEN = 100
            if self.vc_offset > 40 and self.vc_kv_cache[0][0].shape[2] > VC_KV_CACHE_MAX_LEN:
                for i in range(len(self.vc_kv_cache)):
                    new_k = self.vc_kv_cache[i][0][:, :, -VC_KV_CACHE_MAX_LEN:, :]
                    new_v = self.vc_kv_cache[i][1][:, :, -VC_KV_CACHE_MAX_LEN:, :]
                    self.vc_kv_cache[i] = (new_k, new_v)

            mel = x.transpose(1, 2)

            if self.vocoder_cache is not None:
                mel = torch.cat([self.vocoder_cache, mel], dim=-1)
            self.vocoder_cache = mel[:, :, -self.vocoder_overlap:]
            mel = (mel + 1) / 2
            wav = self.vocoder.decode(mel).squeeze()
            wav = wav.detach().cpu().numpy()
            
            if self.last_wav is not None:
                front_wav = wav[:self.vocoder_wav_overlap]
                smooth_front_wav = self.last_wav * self.down_linspace + front_wav * self.up_linspace
                new_wav = np.concatenate([smooth_front_wav, wav[self.vocoder_wav_overlap:-self.vocoder_wav_overlap]], axis=0)
            else:
                new_wav = wav[:-self.vocoder_wav_overlap]
            self.last_wav = wav[-self.vocoder_wav_overlap:]

            return new_wav

    def run(self):
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')

        print("=== Input Device List ===")
        input_devices = []
        for i in range(0, numdevices):
            device_info = p.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                print("Input Device id ", i, " - ", device_info.get('name'))
                input_devices.append(i)

        print("\n=== Output Device List ===")
        output_devices = []
        for i in range(0, numdevices):
            device_info = p.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxOutputChannels') > 0:
                print("Output Device id ", i, " - ", device_info.get('name'))
                output_devices.append(i)

        input_device_id = int(input("Select input device ID: "))
        output_device_id = int(input("Select output device ID: "))

        if input_device_id < 0 or input_device_id not in input_devices:
            input_device_id = p.get_default_input_device_info()['index']
            print(f"Using default input device: {p.get_device_info_by_index(input_device_id)['name']}")

        if output_device_id < 0 or output_device_id not in output_devices:
            output_device_id = p.get_default_output_device_info()['index']
            print(f"Using default output device: {p.get_device_info_by_index(output_device_id)['name']}")

        print("warming up")

        self.in_stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        input_device_index=input_device_id,
                        frames_per_buffer=self.CHUNK)
        self.out_stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=16000,
                        output=True,
                        output_device_index=output_device_id)
        for i in tqdm(range(10)):
            data = self.in_stream.read(self.CHUNK, exception_on_overflow = False)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / (1 << 15)
            Thread(target=self.playaudio, args=(self.out_stream, samples.tobytes())).start()


        self.init_cache()

        i = 0
        while True:

            if i % 50 == 0 and i != 0:
                print("reset!")
                self.reset_cache()
            data = self.in_stream.read(self.CHUNK, exception_on_overflow = False)
            
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / (1 << 15)

            if self.need_extra_data:
                extra_data = self.in_stream.read(720, exception_on_overflow = False)
                extra_samples = np.frombuffer(extra_data, dtype=np.int16).astype(np.float32) / (1 << 15)
                samples = np.concatenate([samples, extra_samples])
                self.need_extra_data = False

            cur_time = time.time()
            vc_wav = self.inference_one_chunk(samples)

            processed_duration = len(samples) / 16000
            Thread(target=self.playaudio, args=(self.out_stream, vc_wav.tobytes())).start()
            print(f"chunk use time{time.time()-cur_time}, chunk size {processed_duration}")
            i += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-path", default="src/runtime/example/test.wav")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)

    runner = VCRunner(args.target_path, steps=args.steps)
    try:
        runner.run()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        runner.cleanup()

if __name__ == "__main__":
    main()