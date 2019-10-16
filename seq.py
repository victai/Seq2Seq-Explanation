import numpy as np
import pandas as pd
import ipdb
import pickle
import argparse
import time
import sys
import random
import math
import logging
from itertools import groupby

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam

device = 'cuda:0'

from model import Encoder, Decoder, LuongAttnDecoderRNN, AttnDecoder

parser = argparse.ArgumentParser()
parser.add_argument('--raw_data', type=str, default='data/lyrics_raw.txt')
parser.add_argument("--create_data", action="store_true", default=False)
parser.add_argument("--train", action="store_true", default=False)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--attn", action="store_true", default=False)
parser.add_argument("--test_data", type=str, default="../data/ta_input.txt")
parser.add_argument("--output_file", type=str, default="result.txt")
parser.add_argument("--model_path", type=str, default="model.pt")
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--bidirectional", type=str, default=False)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--max_assign_cnt", type=int, default=-1)
args = parser.parse_args()
if args.bidirectional == 'False':
    args.bidirectional = False
elif args.bidirectional == 'True':
    args.bidirectional = True


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(message)s')
file_handler = logging.FileHandler('test.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

PAD_TOKEN = '<PAD>' #0
SOS_TOKEN = '<SOS>' #1
EOS_TOKEN = '<EOS>' #2

cut_length = 13 # cut sentences longer than 20 segments
max_seqlen = 26

## Process Data Section

print("Processing Data")
t1 = time.time()

assign_cnt_list = []
Assign = []
def gen_data():
    with open(args.raw_data, 'r') as f:
        data = f.read().split('\n')[:-1]

    all_X = []
    all_Y = []
    for i in range(len(data)-1):
        prev_line = data[i][:cut_length]
        next_line = data[i+1][:cut_length]
        if len(next_line) <= 3: continue
        X = [SOS_TOKEN] + list(prev_line) + [EOS_TOKEN]
        try:
            if args.max_assign_cnt == -1:
                assign_cnt = random.randrange(0, min(math.floor(len(next_line) * 0.5), (max_seqlen-len(prev_line)-1)//2))
            else:
                assign_cnt = min(len(next_line), args.max_assign_cnt)
        except:
            assign_cnt = 0
        assign_cnt_list.append(assign_cnt)
        assign_pos = sorted(random.sample(range(min(len(next_line), 10)), assign_cnt))
        Assign.extend(assign_pos)
        assign_word = np.array(list(next_line))[assign_pos]
        X += [str(assign_cnt)]
        for n, c in zip(assign_pos, assign_word):
            X += [str(n+1), c]
        X += [PAD_TOKEN] * (max_seqlen - len(X))
        all_X.append(X)
        all_Y.append([SOS_TOKEN] + list(next_line) + [EOS_TOKEN] + [PAD_TOKEN] * max(cut_length-len(next_line), 0))
    
    #all_X = all_X[:100000]
    #all_Y = all_Y[:100000]
    with open('all_X.pkl', 'wb') as f:
        pickle.dump(all_X, f)
    with open('all_Y.pkl', 'wb') as f:
        pickle.dump(all_Y, f)

    return all_X, all_Y

if args.create_data:
    all_X, all_Y = gen_data()
else:
    with open('all_X.pkl', 'rb') as f:
        all_X = pickle.load(f)
    with open('all_Y.pkl', 'rb') as f:
        all_Y = pickle.load(f)

all_words = np.unique([i for line in all_X for i in line] + [i for line in all_Y for i in line])
all_words = np.delete(all_words, np.argwhere(all_words == PAD_TOKEN))
all_words = np.delete(all_words, np.argwhere(all_words == SOS_TOKEN))
all_words = np.delete(all_words, np.argwhere(all_words == EOS_TOKEN))
all_words = np.hstack([[PAD_TOKEN, SOS_TOKEN, EOS_TOKEN], all_words])
num_words = len(all_words)
logger.info('num_words: {}'.format(num_words))
if len(assign_cnt_list) > 0:
    logger.info('average assign counts: {}'.format(np.mean(assign_cnt_list)))
word2idx = {k: v for k, v in zip(all_words, range(num_words))}
idx2word = {v: k for k, v in word2idx.items()}

all_X_idx = np.array([[word2idx[word] for word in line] for line in all_X])
all_Y_idx = np.array([[word2idx[word] for word in line] for line in all_Y])

train_cnt = math.floor(len(all_X) * 0.9)
idx = [*range(len(all_X))]
random.shuffle(idx)
train_X = all_X_idx[idx[:train_cnt]]
train_Y = all_Y_idx[idx[:train_cnt]]
valid_X = all_X_idx[idx[train_cnt:]]
valid_Y = all_Y_idx[idx[train_cnt:]]

logger.info("=== Process Data Done ==== Time cost {:.3f} secs".format(time.time() - t1))

logger.info('*** Settings ***')
logger.info('Assign Counts: {}'.format(args.max_assign_cnt))
logger.info('Hidden Dim: {}'.format(args.hidden_dim))
logger.info('Num Layers: {}'.format(args.n_layers))
logger.info('Encoder Bidirectional: {}'.format(args.bidirectional))

## Process Data End


def predict(input_var, tgt_var, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, input_count):
    loss = 0
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_output, encoder_hidden = encoder(input_var)
    decoder_hidden = encoder_hidden
    decoder_input = torch.LongTensor([[word2idx[SOS_TOKEN]]]).repeat(1, input_var.shape[1])
    decoder_input = decoder_input.to(device)
   
    outputs = []
    for i in range(tgt_var.shape[0]):
        if args.attn:
            decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_output)
        else:
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        loss += criterion(decoder_output.squeeze(0), tgt_var[i])
        if args.attn:
            _, top1 = decoder_output.topk(1, dim=1)
            decoder_input = top1.squeeze(-1).unsqueeze(0)
            outputs.append(top1.squeeze(-1).cpu().tolist())
        else:
            _, top1 = decoder_output.topk(1, dim=2)
            decoder_input = top1.squeeze(-1)
            outputs.append(top1.squeeze(0).squeeze(-1).cpu().tolist())

    all_control_cnt = []
    all_hit_cnt = []
    input_var_transpose = input_var.cpu().numpy().T
    outputs = np.array(outputs).T
    input_chinese = []
    output_chinese = []
    for i, line in enumerate(input_var_transpose[:input_count]):
        try:
            input_PAD_position = np.argwhere(line == word2idx[PAD_TOKEN])[0][0]
        except:
            input_PAD_position = None
        output_EOS_position = min(np.argwhere(outputs[i] == word2idx[EOS_TOKEN])[0][0] + 1, len(outputs[i]))
        input_chinese.append(' '.join([idx2word[w] for w in line[:input_PAD_position]]))
        output_chinese.append(' '.join([idx2word[w] for w in outputs[i][:output_EOS_position]]))

    return input_chinese, output_chinese


def train(input_var, tgt_var, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    loss = 0
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_output, encoder_hidden = encoder(input_var)
    decoder_hidden = encoder_hidden
    decoder_input = torch.LongTensor([[word2idx[SOS_TOKEN]]]).repeat(1, input_var.shape[1])
    decoder_input = decoder_input.to(device)
    
    teacher_forcing_ratio = 0.5

    teacher_forcing = False#True if np.random.random() > teacher_forcing_ratio else False
    
    if teacher_forcing:
        for i in range(tgt_var.shape[0]):
            if args.attn:
                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output.squeeze(0), tgt_var[i])
            decoder_input = tgt_var[i].unsqueeze(0)
        #print("encode time:[{:.3f}], en_de time:[{:.3f}], teacher decode time[{:.3f}]".format(b-a, c-b, d_t-c))

    else:
        for i in range(tgt_var.shape[0]):
            if args.attn:
                decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_output)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output.squeeze(0), tgt_var[i])
            if args.attn:
                _, top1 = decoder_output.topk(1, dim=1)
                decoder_input = top1.squeeze(-1).unsqueeze(0)
            else:
                _, top1 = decoder_output.topk(1, dim=2)
                decoder_input = top1.squeeze(-1)
        #print("encode time:[{:.3f}], en_de time:[{:.3f}], normal decode time[{:.3f}]".format(b-a, c-b, d-c))

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / tgt_var.shape[0] / input_var.shape[0]


def valid(input_var, tgt_var, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    loss = 0
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_output, encoder_hidden = encoder(input_var)
    decoder_hidden = encoder_hidden
    decoder_input = torch.LongTensor([[word2idx[SOS_TOKEN]]]).repeat(1, input_var.shape[1])
    decoder_input = decoder_input.to(device)
   
    outputs = []
    for i in range(tgt_var.shape[0]):
        if args.attn:
            decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_output)
        else:
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        loss += criterion(decoder_output.squeeze(0), tgt_var[i])
        if args.attn:
            _, top1 = decoder_output.topk(1, dim=1)
            decoder_input = top1.squeeze(-1).unsqueeze(0)
            outputs.append(top1.squeeze(-1).cpu().tolist())
        else:
            _, top1 = decoder_output.topk(1, dim=2)
            decoder_input = top1.squeeze(-1)
            outputs.append(top1.squeeze(0).squeeze(-1).cpu().tolist())

    all_control_cnt = []
    all_hit_cnt = []
    input_var_transpose = input_var.cpu().numpy().T
    outputs = np.array(outputs).T
    for i, line in enumerate(input_var_transpose):
        EOS_position = np.argwhere(line == word2idx[EOS_TOKEN])[0][0]
        position_control = [list(group) for k, group in groupby(line[EOS_position+1:], lambda x: x == word2idx[PAD_TOKEN]) if not k]
        control_cnt = 0
        hit_cnt = 0
        if position_control != []:
            position_control = position_control[0]
        control_cnt = (len(position_control) - 1) // 2
        for j in range(control_cnt):
            position, word = position_control[1+j*2:1+j*2+2]
            position = int(idx2word[position])
            if outputs[i][position] == word:
                hit_cnt += 1
        all_control_cnt.append(control_cnt)
        all_hit_cnt.append(hit_cnt)


    return loss.item() / tgt_var.shape[0] / input_var.shape[0], all_control_cnt, all_hit_cnt


def load_checkpoint(path, encoder, encoder_optimizer, decoder, decoder_optimizer):
    checkpoint = torch.load(path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']


def main():
    epoch = 1000
    batch_size = 256

    encoder = Encoder(num_words, args.hidden_dim, n_layers=args.n_layers, bidirectional=args.bidirectional).to(device)
    if args.attn:
        decoder = AttnDecoder(args.hidden_dim, num_words, max_seqlen, n_layers=args.n_layers, bidirectional=args.bidirectional).to(device)
    else:
        decoder = Decoder(args.hidden_dim, num_words, n_layers=args.n_layers).to(device)

    if args.train:
        weight = torch.ones(num_words)
        weight[word2idx[PAD_TOKEN]] = 0
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        weight = weight.to(device)
        encoder_optimizer = Adam(encoder.parameters(), lr=0.001)
        decoder_optimizer = Adam(decoder.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=word2idx[PAD_TOKEN])


        np.random.seed(1124)
        order = np.arange(len(train_X))

        best_loss = 1e10
        best_percentage = 0
        best_percentage_epoch = 0
        best_epoch = 0
        start_epoch = 0
        if args.resume:
            start_epoch, best_loss = load_checkpoint(args.model_path, encoder, encoder_optimizer, decoder, decoder_optimizer)
        
        for e in range(start_epoch, start_epoch+epoch):
            if e - best_percentage_epoch > 2: break

            np.random.shuffle(order)
            shuffled_train_X = train_X[order]
            shuffled_train_Y = train_Y[order]
            train_loss = 0
            valid_loss = 0
            
            for b in tqdm(range(int(len(order) // batch_size))):
                batch_x = torch.LongTensor(shuffled_train_X[b*batch_size: (b+1)*batch_size].tolist()).t()
                batch_y = torch.LongTensor(shuffled_train_Y[b*batch_size: (b+1)*batch_size].tolist()).t()

                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                train_loss += train(batch_x, batch_y, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

            train_loss /= b
           
            all_control_cnt, all_hit_cnt = [], []
            for b in range(len(valid_X) // batch_size):
                batch_x = torch.LongTensor(valid_X[b*batch_size: (b+1)*batch_size].tolist()).t()
                batch_y = torch.LongTensor(valid_Y[b*batch_size: (b+1)*batch_size].tolist()).t()
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                val_loss, control_cnt, hit_cnt = valid(batch_x, batch_y, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
                valid_loss += val_loss
                all_control_cnt.extend(control_cnt)
                all_hit_cnt.extend(hit_cnt)
            valid_loss /= b
            all_control_cnt = np.array(all_control_cnt)
            all_hit_cnt = np.array(all_hit_cnt)
            nonzero = all_control_cnt != 0
            all_control_cnt = all_control_cnt[nonzero]
            all_hit_cnt = all_hit_cnt[nonzero]
            percentage = np.mean(all_hit_cnt/all_control_cnt)
            all_correct_percentage = sum(all_hit_cnt == all_control_cnt) / len(all_control_cnt)
            logger.info("epoch {}, train_loss {:.4f}, valid_loss {:.4f}, best_epoch {}, best_loss {:.4f}, control_cnt {}, hit_cnt {}, percentage {:.4f}, all_correct_percentage {:.4f}".format(e, train_loss, valid_loss, best_epoch, best_loss, np.sum(all_control_cnt), np.sum(all_hit_cnt), percentage, all_correct_percentage))

            if percentage > best_percentage:
                best_percentage = percentage
                best_percentage_epoch = e
                torch.save({
                    'encoder_state_dict': encoder.state_dict(),
                    'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                    'epoch': e,
                    'loss': valid_loss,
                    'percentage': best_percentage,
                }, args.model_path)
            
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = e
                torch.save({
                    'encoder_state_dict': encoder.state_dict(),
                    'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                    'epoch': e,
                    'loss': valid_loss
                }, args.model_path)
        
        batch_x = torch.LongTensor(valid_X[:batch_size].tolist()).t()
        batch_y = torch.LongTensor(valid_Y[:batch_size].tolist()).t()
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        input_chinese, output_chinese = predict(batch_x, batch_y, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, 20)
        
        logger.info('*** Results ***')
        logger.info('Best Hit Accuracy: {}'.format(best_percentage))
        logger.info('Best Hit Accuracy Epoch: {}'.format(best_percentage_epoch))
        for inp, out in zip(input_chinese, output_chinese):
            logger.info('{}\t||\t{}'.format(inp, out))
        logger.info(encoder)
        logger.info(decoder)
        logger.info('\n\n' + '='*100 + '\n\n')

    else:
        print(encoder)
        print(decoder)


    #predict(args.test_data, encoder, decoder)
    #submit()

if __name__ == '__main__':
    main()
    
