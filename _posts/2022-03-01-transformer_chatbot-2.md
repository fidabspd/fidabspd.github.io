---
layout: post
title: Transformer - Chatbot 만들기 (2)
tags: [Chatbot, NLP, Transformer, PyTorch]
excerpt_separator: <!--more-->
use_math: true
---

[지난 포스팅](https://fidabspd.github.io/2022/02/23/transformer_chatbot-1.html)에서 PyTorch로 만든 Transformer를 훈련하고, 이를 이용하여 Chatbot을 만들어보자.<!--more-->

## 목차

1. [Chatbot 만들기 (1)](https://fidabspd.github.io/2022/02/23/transformer_chatbot-1.html)
1. [**Chatbot 만들기 (2)**](https://fidabspd.github.io/2022/03/01/transformer_chatbot-2.html)
1. [Chatbot 만들기 (3)](https://fidabspd.github.io/2022/03/02/transformer_chatbot-3.html)

## 원본 코드 ➞ [<span style="color:#AC1538">CODE (GitHub)</span>](https://github.com/fidabspd/toy/blob/master/chatbot/codes)

## Data

데이터는 [songys](https://github.com/songys)님의 [ChatbotData](https://github.com/songys/Chatbot_data/blob/master/ChatbotData.csv)를 사용하였다.  

데이터는 다음과 같다.

```
12시 땡!	하루가 또 가네요.
1지망 학교 떨어졌어	위로해 드립니다.
3박4일 놀러가고 싶다	여행은 언제나 좋죠.
3박4일 정도 놀러가고 싶다	여행은 언제나 좋죠.
PPL 심하네	눈살이 찌푸려지죠.
SD카드 망가졌어	다시 새로 사는 게 마음 편해요.
SD카드 안돼	다시 새로 사는 게 마음 편해요.
SNS 맞팔 왜 안하지ㅠㅠ	잘 모르고 있을 수도 있어요.
SNS 시간낭비인 거 아는데 매일 하는 중	시간을 정하고 해보세요.
SNS 시간낭비인데 자꾸 보게됨	시간을 정하고 해보세요.
...
```

대화형식의 말과 대답으로 쌍지어져있으며 11823개의 대화 쌍이 존재한다.

## CODE

코드가 너무 길기 때문에 핵심이 되는 부분만 짚어보도록 하자.

### Data Preprocessing

Transformer의 학습과 평가에 활용할 수 있도록 전처리를 수행한다.

```python
import re
import numpy as np


def add_space(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence


def to_tokens(sentence, tokenizer, to_ids=True):
    if to_ids:
        tokens = tokenizer.encode(sentence).ids
        vocab_size = tokenizer.get_vocab_size()
        start_token, end_token = vocab_size, vocab_size+1
        tokens = [start_token]+tokens+[end_token]
    else:
        tokens = tokenizer.encode(sentence).tokens
    return tokens


def pad_seq(seq, tokenizer, max_seq_len):
    pad_token = tokenizer.encode('[PAD]').ids[0]
    padded_seq = seq+[pad_token]*(max_seq_len-len(seq))
    return padded_seq


def preprocess_sentence(sentence, tokenizer, max_seq_len):
    sentence = add_space(sentence)
    sentence = to_tokens(sentence, tokenizer)
    sentence = pad_seq(sentence, tokenizer, max_seq_len)
    return sentence


def preprocess_sentences(sentences, tokenizer, max_seq_len):
    prep =  list(map(
        lambda sentence:
            preprocess_sentence(sentence, tokenizer, max_seq_len),
        sentences
    ))
    return np.array(prep)
```

- `add_space`: 기호 앞 뒤에 공백을 추가한다.
- `to_tokens`: 문장을 token으로 분리한다. 사용하는 tokenizer에 대해서는 뒤에서 설명한다.
- `pad_seq`: 길이에 맞춰 padding을 수행한다.
- `preprocess_sentence`: 위 세개를 모두 수행한다.
- `preprocess_sentences`: setence 리스트에 대해 각각 `preprocess_sentence`를 수행한다.

### Train Chatbot

Chatbot 데이터를 이용하여 Transformer를 훈련하고 모델을 저장한다.

### Libraries

```python
import os
import time
import math
import argparse
import pickle

from tokenizers import BertWordPieceTokenizer

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from chatbot_data_preprocessing import *
from transformer_torch import *
```

### Train One Epoch

```python
def train_one_epoch(model, dl, optimizer, criterion, clip, device, n_check=5):

    n_data = len(dl.dataset)
    n_batch = len(dl)
    batch_size = dl.batch_size
    if n_check < 0:
        print('n_check must be larger than 0. Adjust `n_check = 0`')
        n_check = 0
    if n_batch < n_check:
        print(f'n_check should be smaller than n_batch. Adjust `n_check = {n_batch}`')
        n_check = n_batch
    if n_check:
        check = [int(n_batch/n_check*(i+1)) for i in range(n_check)]
    train_loss = 0

    model.train()
    for b, (inp, tar) in enumerate(dl):
        inp, tar = inp.to(device), tar.to(device)

        outputs, _ = model(inp, tar[:,:-1])
        # 번역기로 예를들어 말하자면
        # inp: [<sos>, hi, <eos>, <pad>, <pad>], tar: [<sos>, 안녕, <eos>, <pad>, <pad>, <pad>]을 이용해
        # [안녕, <eos>, <pad>, <pad>, <pad>, <pad>]의 예측값을 만들어내야한다.

        output_dim = outputs.shape[-1]
        outputs = outputs.contiguous().view(-1, output_dim)
        tar = tar[:,1:].contiguous().view(-1)  # loss 계산할 정답으로 쓰일 `tar`는 <sos> 토큰 제거
        loss = criterion(outputs, tar)
        train_loss += loss.item()/n_data

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if n_check and b+1 in check:
            n_data_check = b*batch_size + len(inp)
            train_loss_check = train_loss*n_data/n_data_check
            print(f'loss: {train_loss_check:>10f}  [{n_data_check:>5d}/{n_data:>5d}]')

    return train_loss
```

`train_one_epoch`은 한번의 epoch에 대한 훈련을 수행한다.  

눈여겨볼 부분은 주석이 붙은 부분들이다.  

- loss 계산에 정답으로 쓰일 `tar`는 `<sos>`토큰을 제거한다.  
[Chatbot 만들기 (1)](https://fidabspd.github.io/2022/02/23/transformer_chatbot-1.html)에서 말 했듯이 Transformer는 다음 토큰을 예측하는데 쓰는 모델이다. `<sos>`를 예측해야하는 이전 토큰은 당연히 없다.
- 모델 input으로 들어가는 `tar`는 sequence의 마지막을 뗀다.  
Transformer의 최종 output의 shape은 `[batch_size, query_len(tar), decoder vocab_size]`이다.  
**모델 input으로 들어가는 `tar`**의 sequence 길이와, **loss 계산에 정답으로 쓰일 `tar`**는 길이가 같아야한다.

### Evaluate

```python
def evaluate(model, dl, criterion, device):
    n_data = len(dl.dataset)
    
    valid_loss = 0

    model.eval()
    with torch.no_grad():
        for inp, tar in dl:
            inp, tar = inp.to(device), tar.to(device)
            outputs, _ = model(inp, tar[:,:-1])

            output_dim = outputs.shape[-1]

            outputs = outputs.contiguous().view(-1, output_dim)
            tar = tar[:,1:].contiguous().view(-1)
            loss = criterion(outputs, tar)

            valid_loss += loss.item()/n_data

    return valid_loss
```

`evaluate`은 모델을 평가하는데 사용된다.  
`train_one_epoch`과 마찬가지로, loss 계산에 정답으로 쓰일 `tar`는 `<sos>`토큰을 제거하고, 모델 input으로 들어가는 `tar`는 sequence의 마지막을 뗀다.

### Train

```python
def train(model, n_epochs, es_patience, train_dl, valid_dl, optimizer,
          criterion, clip, device, model_path, train_log_path, model_name='chatbot'):
    if train_log_path is not None:
        writer = SummaryWriter(train_log_path)
    best_valid_loss = float('inf')
    best_epoch = 0

    for epoch in range(n_epochs):
        start_time = time.time()

        print('-'*30, f'\nEpoch: {epoch+1:02}', sep='')
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, clip, device)
        if train_log_path is not None:
            writer.add_scalar('train loss', train_loss, epoch)
        if valid_dl is not None:
            valid_loss = evaluate(model, valid_dl, criterion, device)
            if train_log_path is not None:
                writer.add_scalar('valid loss', valid_loss, epoch)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_dl is not None:
            if valid_loss < best_valid_loss:
                best_epoch = epoch
                print('Best!')
                best_valid_loss = valid_loss
                torch.save(model, model_path+model_name+'.pt')

        print(f'Train Loss: {train_loss:.3f}\nEpoch Time: {epoch_mins}m {epoch_secs}s')
        if valid_dl is not None:
            print(f'Validation Loss: {valid_loss:.3f}')

            if epoch-best_epoch >= es_patience:
                print(f'\nBest Epoch: {best_epoch+1:02}')
                print(f'\tBest Train Loss: {train_loss:.3f}')
                print(f'\tBest Validation Loss: {valid_loss:.3f}')
                break
    
    if train_log_path is not None:
        writer.close()
    if valid_dl is None:
        torch.save(model, model_path+model_name+'.pt')
```

`train`은 epoch을 반복하며 모델을 훈련한다.  

`valid_dl`이 없다면 `n_epochs`만큼 반복하여 훈련한다.
`valid_dl`이 있다면 `valid_loss` 기준 best 모델만을 저장하며 `es_patience`만큼 새로운 `best_valid_loss`가 갱신되지 않으면 훈련을 멈춘다.  

Tensorboard를 이용하여 loss를 기록한다.

훈련하면서 깨달은 점이 있다.  
데이터가 적어서인지 `valid_loss`가 쉽게 줄어들지 않는다. `valid_loss`기준으로 훈련을 멈추게 되면 챗봇이 제대로 작동하지 않는다.  
같은말만 계속 반복하는 등 문장 자체가 언어적으로 이해할 수 없는 문장을 생성한다.  

대신 `valid_dl`을 주지 않고 train set 만으로 과적합을 하게 되면 대화는 되지 않을지언정 언어적으로 말이 되는 문장을 준다.  
따라서 `valid_dl`은 사용하지 않고 과적합시켰다.

또한 챗봇을 제대로 만들 생각이라면 loss에 대한 고찰이 필요할 것으로 보인다.  
예를들어 **100원만 빌려줘**라는 말에  
**그래**, **싫어**, **100원이 없어**, **100원이 없니?** 등 나올 수 있는 답변은 무궁무진하다. 물론 이는 데이터셋으로 커버할 수 있긴 하다. 그럼 이 데이터를 모두 훈련했다고 가정하고 다음 답변이 나온다면 어떨까?  
**음... 그래!**

**그래** 역시 답변으로 학습했기 때문에 **음... 그래!**는 loss가 작을까? 절대 그렇지 않다.  
이런 경우와 같이 옆으로 한칸씩 이동한 경우에도 사실상 정답이지만 현재 사용한 loss에서는 이를 전혀 고려하지 못하고 있다. (`CrossEntropyLoss`를 사용했다.)  

챗봇을 제대로 만들고 싶다면 이런 loss에 대한 고찰 역시 필요할 것으로 보인다.

훈련은 다음과 같이 진행된다.

```
------------------------------
Epoch: 01
loss:  47.228932  [ 2368/11823]
loss:  43.583458  [ 4736/11823]
loss:  41.757317  [ 7104/11823]
loss:  40.540467  [ 9472/11823]
loss:  39.534571  [11823/11823]
Train Loss: 39.535
Epoch Time: 0m 9s
------------------------------
```

### Main

```python
def main(args):

    DATA_PATH = args.data_path
    FILE_NAME = args.file_name
    GRAPH_LOG_PATH = args.graph_log_path
    TOKENIZER_PATH = args.tokenizer_path
    TOKENIZER_NAME = args.tokenizer_name
    MODEL_PATH = args.model_path
    MODEL_NAME = args.model_name
    TRAIN_LOG_PATH = args.train_log_path

    QUE_MAX_SEQ_LEN = args.que_max_seq_len
    ANS_MAX_SEQ_LEN = args.ans_max_seq_len
    N_LAYERS = args.n_layers
    HIDDEN_DIM = args.hidden_dim
    N_HEADS = args.n_heads
    PF_DIM = args.pf_dim
    DROPOUT_RATIO = args.dropout_ratio

    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    CLIP = args.clip
    N_EPOCHS = args.n_epochs
    ES_PATIENCE = args.es_patience
    VALIDATE = args.validate

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')

    # Load data
    questions = []
    answers = []
    f = open(DATA_PATH+FILE_NAME+'.txt', 'r')
    while True:
        line = f.readline()
        if not line:
            break
        question, answer = line.split('\t')
        questions.append(question)
        answers.append(answer)
    f.close()

    # Train tokenizer
    tokenizer = BertWordPieceTokenizer(strip_accents=False, lowercase=False)
    tokenizer.train(
        files = DATA_PATH+FILE_NAME+'.txt',
        vocab_size = 32000,
        min_frequency = 3,
        limit_alphabet = 6000
    )
    if not os.path.exists(TOKENIZER_PATH):
        os.makedirs(TOKENIZER_PATH)
    with open(TOKENIZER_PATH+TOKENIZER_NAME+'.pickle', 'wb') as f:
        pickle.dump(tokenizer, f)
    vocab_size = tokenizer.get_vocab_size()
    PAD_IDX = tokenizer.encode('[PAD]').ids[0]
    INPUT_DIM = vocab_size+2  # start_token, end_token
    OUTPUT_DIM = vocab_size+2

    # Preprocess data
    questions_prep = preprocess_sentences(questions, tokenizer, QUE_MAX_SEQ_LEN)
    answers_prep = preprocess_sentences(answers, tokenizer, ANS_MAX_SEQ_LEN)

    class ChatBotDataset(Dataset):
        def __init__(self, questions, answers):
            assert len(questions) == len(answers)
            self.questions = questions
            self.answers = answers
            
        def __len__(self):
            return len(self.questions)
        
        def __getitem__(self, idx):
            question, answer = self.questions[idx], self.answers[idx]
            return question, answer

    if VALIDATE:
        train_q, valid_q = questions_prep[:-3000], questions_prep[-3000:]
        train_a, valid_a = answers_prep[:-3000], answers_prep[-3000:]
        valid_ds = ChatBotDataset(valid_q, valid_a)
        valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE)
    else:
        train_q = questions_prep
        train_a = answers_prep

    train_ds = ChatBotDataset(train_q, train_a)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Set model
    transformer = Transformer(
        INPUT_DIM, OUTPUT_DIM, N_LAYERS, HIDDEN_DIM, N_HEADS, PF_DIM,
        QUE_MAX_SEQ_LEN, ANS_MAX_SEQ_LEN, PAD_IDX, DROPOUT_RATIO, device
    ).to(device)

    print(f'# of trainable parameters: {count_parameters(transformer):,}')
    transformer.apply(initialize_weights)

    inp, tar = iter(train_dl).next()
    inp, tar = inp.to(device), tar.to(device)
    create_tensorboard_graph(transformer, (inp, tar), GRAPH_LOG_PATH)

    # Train model
    optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_IDX)

    if not VALIDATE:
        valid_dl = None
    train(
        transformer, N_EPOCHS, ES_PATIENCE, train_dl, valid_dl,
        optimizer, criterion, CLIP, device, MODEL_PATH, TRAIN_LOG_PATH, MODEL_NAME
    )

    if VALIDATE:
        transformer = torch.load(MODEL_PATH+MODEL_NAME+'.pt')
        valid_loss = evaluate(transformer, valid_dl, criterion, device)
        print(f'Valid Loss: {valid_loss:.3f} | Valid PPL: {math.exp(valid_loss):.3f}')


if __name__ == '__main__':

    args = parse_args()
    if args is None:
        exit()

    main(args)
```

main함수이다.  

tokenizer는 `BertWordPieceTokenizer`를 사용한다. 성능을 고려한것은 아니고 가장 사용하기 간편하다고 생각했다.

optimizer는 `Adam`을 사용한다.

loss는 `CrossEntropyLoss`를 사용한다. `ignore_index`를 통해 `<pad>`를 loss 계산에 포함하지 않을 수 있다.

만약 논문을 읽어봤다면 구현되지 않은 것이 하나 보일 것이다.  
[Attention Is All You Need](https://arxiv.org/abs/1706.03762)에서는 Optimizer에 대한 learning rate scheduler를 제안한다.  
하지만 이는 그냥 생략했다.

위 코드가 있는 ```train_chatbot.py```를 실행하면 훈련을 수행하고 transformer 모델과 tokenizer를 저장한다.  

이렇게 훈련을 마쳤다. 

분량이 또 다시 길어져 Chatbot을 만들고 실행하는 것은 3부에서 계속하도록 하자.

## 목차

1. [Chatbot 만들기 (1)](https://fidabspd.github.io/2022/02/23/transformer_chatbot-1.html)
1. [**Chatbot 만들기 (2)**](https://fidabspd.github.io/2022/03/01/transformer_chatbot-2.html)
1. [Chatbot 만들기 (3)](https://fidabspd.github.io/2022/03/02/transformer_chatbot-3.html)

## 원본 코드 ➞ [<span style="color:#AC1538">CODE (GitHub)</span>](https://github.com/fidabspd/toy/blob/master/chatbot/codes)
