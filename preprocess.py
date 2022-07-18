from msilib import make_id
import os
import pdb
import pickle
import argparse
import numpy as np
from tqdm import tqdm, trange
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from transformers import ElectraTokenizer

def make_id_file(task, tokenizer):
    def make_data_strings(file_name):
        data_strings = []
        with open(os.path.join(file_name), 'r', encoding='utf-8') as f:
            id_file_data = [tokenizer.encode(line.lower()) for line in f.readlines()]
        for item in id_file_data:
            data_strings.append(' '.join([str(k) for k in item]))
        return data_strings

    print('it will take some times...')
    train_pos = make_data_strings('sentiment.train.1')
    train_neg = make_data_strings('sentiment.train.0')
    dev_pos = make_data_strings('sentiment.dev.1')
    dev_neg = make_data_strings('sentiment.dev.0')

    print('make id file finished!')
    return train_pos, train_neg, dev_pos, dev_neg


tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
results = make_id_file('yelp', tokenizer)
filenames = ['train_pos', 'train_neg', 'val_pos', 'val_neg']

for i, v in enumerate(results):
    filename = f'./data/{filenames[i]}'
    with open(filename, 'wb') as f:
        pickle.dump(v, f)