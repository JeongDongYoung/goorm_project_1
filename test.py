import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from tqdm import tqdm, trange

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AlbertForSequenceClassification,
    ElectraForSequenceClassification,
    ElectraTokenizer,
    AutoConfig,
    AdamW,
    get_linear_schedule_with_warmup
)

import pickle
import argparse

class SentimentTestDataset(object):
    def __init__(self, tokenizer, test):
        self.tokenizer = tokenizer
        self.data = []

        for sent in test:
            self.data += [self._cast_to_int(sent.strip().split())]

    def _cast_to_int(self, sample):
        return [int(word_id) for word_id in sample]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return np.array(sample)


def collate_fn_style_test(samples):
    input_ids = samples
    max_len = max(len(input_id) for input_id in input_ids)
    sorted_indices = range(len(input_ids))
    attention_mask = torch.tensor(
        [[1] * len(input_ids[index]) + [0] * (max_len - len(input_ids[index])) for index in
         sorted_indices])
    input_ids = pad_sequence([torch.tensor(input_ids[index]) for index in sorted_indices],
                             batch_first=True)
    token_type_ids = torch.tensor([[0] * len(input_ids[index]) for index in sorted_indices])
    position_ids = torch.tensor([list(range(len(input_ids[index]))) for index in sorted_indices])

    return input_ids, attention_mask, token_type_ids, position_ids

def make_id_file_test(tokenizer, test_dataset):
    data_strings = []
    id_file_data = [tokenizer.encode(sent.lower()) for sent in test_dataset]
    for item in id_file_data:
        data_strings.append(' '.join([str(k) for k in item]))
    return data_strings

def test():
    tokenizer = ElectraForSequenceClassification.from_pretrained('google/electra-base-discriminator')
    
    test_df = pd.read_csv('./data/test_no_label.csv')
    test_dataset = test_df['Id']
    
    test = make_id_file_test(tokenizer, test_dataset)
    
    test_dataset = SentimentTestDataset(tokenizer, test)
    
    test_batch_size = 32
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=False,
                                              collate_fn=collate_fn_style_test,
                                              num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ElectraForSequenceClassification.from_pretrained('google/electra-base-discriminator')
    
    model.load_state_dict(torch.load('./pytorch_model.bin'))
    model.to(device)

    with torch.no_grad():
        model.eval()
        predictions = []
        for input_ids, attention_mask, token_type_ids, position_ids in tqdm(test_loader,
                                                                        desc='Test',
                                                                        position=1,
                                                                        leave=None):

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            position_ids = position_ids.to(device)

            output = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           position_ids=position_ids)

            logits = output.logits
            batch_predictions = [0 if example[0] > example[1] else 1 for example in logits]
            predictions += batch_predictions        
        
if __name__ == '__main__':
    test()