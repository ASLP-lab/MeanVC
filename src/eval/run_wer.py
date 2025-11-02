import sys, os
from tqdm import tqdm
from jiwer import compute_measures
from zhon.hanzi import punctuation
import string
import numpy as np
import soundfile as sf
import scipy
import zhconv
from funasr import AutoModel
import glob

punctuation_all = punctuation + string.punctuation

# # wav_res_text_path = sys.argv[1]
# wav_dir = sys.argv[1]
# res_path = sys.argv[2]
# lang = sys.argv[3] # zh or en
# # device = "cuda:0"
# device = sys.argv[4]
# text = sys.argv[5]

def load_zh_model():
    model = AutoModel(model="paraformer-zh")
    return model

def process_one(hypo, truth):
    for x in punctuation_all:
        if x == '\'':
            continue
        truth = truth.replace(x, '')
        hypo = hypo.replace(x, '')

    truth = truth.replace('  ', ' ')
    hypo = hypo.replace('  ', ' ')


    truth = " ".join([x for x in truth])
    hypo = " ".join([x for x in hypo])

    measures = compute_measures(truth, hypo)
    wer = measures["wer"]
    return wer
