import numpy as np
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

class SentimentDataset(object):
    def __init__(self, tokenizer, pos, neg):
        self.tokenizer = tokenizer
        self.data = []
        self.label = []

        for pos_sent in pos:
            self.data += [self._cast_to_int(pos_sent.strip().split())]
            self.label += [[1]]
        for neg_sent in neg:
            self.data += [self._cast_to_int(neg_sent.strip().split())]
            self.label += [[0]]

    def _cast_to_int(self, sample):
        return [int(word_id) for word_id in sample]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return np.array(sample), np.array(self.label[index])

train_dataset = SentimentDataset(tokenizer, train_pos, train_neg)
dev_dataset = SentimentDataset(tokenizer, val_pos, val_neg)

def collate_fn_style(samples): #sample이 batch단위로 여러 개
    input_ids, labels = zip(*samples)
    max_len = max(len(input_id) for input_id in input_ids)
    sorted_indices = np.argsort([len(input_id) for input_id in input_ids])[::-1]

    attention_mask = torch.tensor(
        [[1] * len(input_ids[index]) + [0] * (max_len - len(input_ids[index])) for index in
         sorted_indices])

    input_ids = pad_sequence([torch.tensor(input_ids[index]) for index in sorted_indices],
                             batch_first=True)
    token_type_ids = torch.tensor([[0] *  len(input_ids[index]) for index in sorted_indices])
    position_ids = torch.tensor([list(range(len(input_ids[index]))) for index in sorted_indices])
    labels = torch.tensor(np.stack(labels, axis=0)[sorted_indices])

    return input_ids, attention_mask, token_type_ids, position_ids, labels

def train(args):
    tokenizer = ElectraForSequenceClassification.from_pretrained('google/electra-base-discriminator')

    filenames = ['train_pos', 'train_neg', 'dev_pos', 'dev_neg']
    temp = []
    for filename in filenames:
        with open(f'./data/{filename}', 'rb') as f:
            temp.append(pickle.load(f))
        
    train_pos, train_neg, dev_pos, dev_neg = temp[0], temp[1], temp[2], temp[3]

    train_dataset = SentimentDataset(tokenizer, train_pos, train_neg)
    dev_dataset = SentimentDataset(tokenizer, dev_pos, dev_neg)
    
    train_batch_size = args.train_bs
    eval_batch_size = args.eval_bs
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True,
                                               collate_fn=collate_fn_style,
                                               pin_memory=True,
                                               num_workers=0)
    dev_loader = torch.utils.data.DataLoader(dev_dataset,
                                             batch_size=eval_batch_size,
                                             shuffle=True,
                                             collate_fn=collate_fn_style,
                                             pin_memory=True,
                                             num_workers=0)

    random_seed=42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AlbertForSequenceClassification.from_pretrained('google/electra-base-discriminator')
    model.to(device)



    model.train()
    learning_rate = args.lr
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    def compute_acc(predictions, target_labels):
        return (np.array(predictions) == np.array(target_labels)).mean()
    
    train_epoch = args.n_epoch
    total_training_steps = train_epoch * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_training_steps=total_training_steps,
                                                num_warmup_steps=200)

    lowest_valid_loss = 0.061
    for epoch in range(train_epoch):
        with tqdm(train_loader, unit="batch") as tepoch: # tepcoh 각각 batch에 대해서 아래 4개
            losses = []
            for iteration, (input_ids, attention_mask, token_type_ids, position_ids, labels) in enumerate(tepoch):
               tepoch.set_description(f"Epoch {epoch}")
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                position_ids = position_ids.to(device)
                labels = labels.to(device, dtype=torch.long)

                optimizer.zero_grad()

                output = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           position_ids=position_ids,
                           labels=labels)

                loss = output.loss
                losses.append(loss.item())
                loss.backward()            
                
                optimizer.step()
                scheduler.step()

                tepoch.set_postfix(loss=np.mean(losses))                     
                if iteration != 0 and iteration % int(len(train_loader) / 20) == 0: # 20마다 체크
                    with torch.no_grad():
                        model.eval()
                        valid_losses = []
                        predictions = []
                        target_labels = []
                        for input_ids, attention_mask, token_type_ids, position_ids, labels in tqdm(dev_loader,
                                                                                                desc='Eval',
                                                                                                position=1,
                                                                                                leave=None):
                            input_ids = input_ids.to(device)
                            attention_mask = attention_mask.to(device)
                            token_type_ids = token_type_ids.to(device)
                            position_ids = position_ids.to(device)
                            labels = labels.to(device, dtype=torch.long)

                            output = model(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           token_type_ids=token_type_ids,
                                           position_ids=position_ids,
                                           labels=labels)

                            logits = output.logits
                            loss = output.loss
                            valid_losses.append(loss.item())
                  

                            batch_predictions = [0 if example[0] > example[1] else 1 for example in logits]
                            batch_labels = [int(example) for example in labels]

                            predictions += batch_predictions
                            target_labels += batch_labels

                    acc = compute_acc(predictions, target_labels)
                    valid_loss = sum(valid_losses) / len(valid_losses)
                
                
                    losses = []
                    model.train()

                    if lowest_valid_loss > valid_loss:
                        print('Acc for model which have lower valid loss: ', acc, 'valid loss: ', valid_loss)
                        torch.save(model.state_dict(), "./pytorch_model_project1_.bin")
                        lowest_valid_loss = valid_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train_bs', type=int, default=32)
    parser.add_argument('--eval_bs', type=int, default=64)
    parser.add_argument('--lr', type=float, default=32)
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--proj_name', type=str, default='batch_')
    
    args = parser.parse_args()
    train(args)