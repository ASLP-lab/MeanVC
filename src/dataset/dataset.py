import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from tqdm import tqdm
import traceback
import sys
import json
import time

MAX_INT = sys.maxsize

torch.multiprocessing.set_start_method('spawn', force=True)
torch.multiprocessing.current_process().authkey = b'my_authkey'



class DiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, file_lst, feature_list, additional_feature_list=[], feature_pad_values=[], max_len=1000,):
        self.file_lst = file_lst
        self.max_len = max_len
        self.feature_list = feature_list
        self.additional_feature_list = additional_feature_list
        self.pad_values = [float(x) for x in feature_pad_values]


        random.seed(42)

    @staticmethod
    def init_data(fileid_list_path):
        file_lst = []
        with open(fileid_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    file_lst.append(line)
        
        random.seed(42)
        random.shuffle(file_lst)
        return file_lst

    @staticmethod
    def load_npy(path):
        try:
            data = np.load(path)
            return data
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    
    def get_concat_prompts(self, prompt_mel_paths, min_frames=2000):
        candidates = prompt_mel_paths.copy()
        concat_prompt = None
        
        while True:
            if not candidates:
                raise ValueError("Not enough prompts to reach target length")
            
            mel_path = random.choice(candidates)
            candidates.remove(mel_path)
            
            mel = self.load_npy(mel_path)
            if mel is None:
                continue
            
            if mel.shape[1] != 80:
                mel = mel.T
            
            if concat_prompt is None:
                concat_prompt = mel
            else:
                concat_prompt = np.concatenate((concat_prompt, mel), axis=0)
            
            if concat_prompt.shape[0] >= min_frames:
                break
            
            if len(candidates) == 0:
                candidates = prompt_mel_paths.copy()
        
        return concat_prompt
    
    
    def get_triple(self, item):
        """
        format : utt|bn_path|mel_path|xvector_path|prompt_mel_path1|prompt_mel_path2|...
        """
        features = dict()
        parts = item.split("|")
        
        if len(parts) < 4:
            raise ValueError(f"Invalid data format: {item}")
        
        name = parts[0]
        bn_path = parts[1]
        mel_path = parts[2]
        xvector_path = parts[3]
        prompt_mel_paths = parts[4:]

        # load bn
        bn = self.load_npy(bn_path)
        if bn is None:
            raise ValueError(f"Failed to load bn from {bn_path}")
        bn = torch.from_numpy(bn)
        bn = bn.unsqueeze(0).transpose(1, 2)
        bn = torch.nn.functional.interpolate(bn, size=int(bn.shape[2] * 4), mode='linear', align_corners=True)
        bn = bn.transpose(1, 2)
        bn = bn.squeeze(0).numpy()
        features['bn'] = bn
        
        # load mel
        mel = self.load_npy(mel_path)
        if mel is None:
            raise ValueError(f"Failed to load mel from {mel_path}")
        if mel.shape[1] != 80:
            mel = mel.T
        if mel.min().item() < -1.5:
            mel = mel / 4 
        features['mel'] = mel
        
        # load xvector
        xvector = self.load_npy(xvector_path)
        if xvector is None:
            raise ValueError(f"Failed to load xvector from {xvector_path}")
        xvector = np.squeeze(xvector, axis=None)
        features['xvector'] = xvector
        
        min_len = min(bn.shape[0], mel.shape[0])
        min_len = min(self.max_len, min_len)
        features['bn'] = features['bn'][:min_len]
        features['mel'] = features['mel'][:min_len]
        
        # load prompt mels
        if len(prompt_mel_paths) > 0:
            concat_prompt = self.get_concat_prompts(prompt_mel_paths, 2000)
            
            total_frames = concat_prompt.shape[0]
            target_len = 2000   # 2000 frame
            if total_frames >= target_len:
                start_idx = np.random.randint(0, total_frames - target_len + 1)
                target_prompt = concat_prompt[start_idx : start_idx + target_len]
            else:
                target_prompt = concat_prompt
        
            features['prompt'] = target_prompt
        else:
            raise ValueError(f"Failed to load prompt mels from {prompt_mel_paths}")
        
        features['inputs_length'] = min_len
        features['filename'] = name
        
        return features
    
    def __getitem__(self, index):
        
        idx = index
        while True:
            try:
                item = self.get_triple(self.file_lst[idx])
                
                return item
            except Exception as e:
                idx = random.randint(0, self.__len__() - 1)
                continue

    def __len__(self):
        return len(self.file_lst)

    def pad_feature(self, feature, seq_len, pad_value):
        if pad_value is None:
            return feature
        paded_feature = np.full(
            [seq_len, feature.shape[1]], fill_value=pad_value, dtype=np.float32
        )
        paded_feature[: len(feature)] = feature
        return paded_feature
    
    def pad_feature_aduio(self, audio, seq_len, pad_value):
        if pad_value is None:
            return audio
        paded_feature = np.full(
            seq_len, fill_value=pad_value, dtype=np.float32
        )
        paded_feature[: len(audio)] = audio
        return paded_feature
    

    def custom_collate_fn(self, batch):
        max_len = max([b[list(b.keys())[0]].shape[0] for b in batch])
        feature_dict = dict()

        for feature_name, pad_value in zip(self.feature_list, self.pad_values):
            if feature_name == 'audio':
                feature_pad_list = [self.pad_feature_aduio(b[feature_name], max_len*160, pad_value) for b in batch]
                feature = torch.as_tensor(np.asarray(feature_pad_list))
            elif feature_name != 'xvector':
                feature_pad_list = [self.pad_feature(b[feature_name], max_len, pad_value) for b in batch]
                feature = torch.as_tensor(np.asarray(feature_pad_list))
            else:
                feature = torch.as_tensor(np.asarray([b[feature_name] for b in batch]))
            feature_dict[feature_name] = feature

        for feature_name in self.additional_feature_list:
            feature_dict[feature_name] = torch.as_tensor(np.asarray([b[feature_name] for b in batch]))

        feature_dict['filename'] = [b['filename'] for b in batch]

        return feature_dict


if __name__ == "__main__":
    

    file_lst = DiffusionDataset.init_data("emilia.txt")
    
    ldd = DiffusionDataset(file_lst,
                            feature_list=["bn", "mel", "xvector"],
                            additional_feature_list=["inputs_length", "prompt"],
                            feature_pad_values=[0.0, -1.0, 0.0],
                            max_len=1000)
    
    import pdb; pdb.set_trace()
    hhh = ldd[0]
    

    import time
    
    start_time = time.time()
    dataloader = DataLoader(
        dataset=ldd,
        batch_size=4,
        shuffle=True,
        num_workers=1, 
        pin_memory=True,   
        collate_fn=ldd.custom_collate_fn
    )
    

    cnt = 0
    feature_list=["bn", "mel", "xvector"]
    additional_feature_list=["inputs_length", "prompt"]
    for batch in tqdm(dataloader):
        features = {}
        for feature_name in feature_list + additional_feature_list:
            features[feature_name] = batch[feature_name]
        
        import pdb; pdb.set_trace()

    print(f"cnt: {cnt}")
