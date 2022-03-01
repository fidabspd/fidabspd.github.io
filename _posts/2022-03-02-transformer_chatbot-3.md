---
layout: post
title: Transformer - Chatbot 만들기 (3)
tags: [Chatbot, NLP, Transformer, PyTorch]
excerpt_separator: <!--more-->
use_math: true
---

지난 포스팅에서 훈련을 마친 Transformer로 Chatbot을 만들어보자.<!--more-->

## CODE

### Libraries

```python
import argparse
import pickle
import matplotlib.pyplot as plt
import torch
from chatbot_data_preprocessing import *
```

### Chatbot

```python
class Chatbot():

    def __init__(self, transformer, tokenizer, device):
        self.transformer = transformer.to(device)
        self.transformer.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.call_qna = False

    def qna(self, question):

        ans_seq_len = self.transformer.out_seq_len

        vocab_size = self.tokenizer.get_vocab_size()
        start_token, end_token = vocab_size, vocab_size+1
        
        question_tokens = to_tokens(question, self.tokenizer)
        question_tokens = torch.LongTensor(question_tokens).unsqueeze(0).to(self.device)
        question_mask = self.transformer.create_padding_mask(question_tokens)
        with torch.no_grad():
            question_encd = self.transformer.encoder(question_tokens, question_mask)

        output_tokens = [start_token]

        for _ in range(ans_seq_len):
            target_tokens = torch.LongTensor(output_tokens).unsqueeze(0).to(self.device)

            target_mask = self.transformer.create_padding_mask(target_tokens, True)
            with torch.no_grad():
                output, attention = self.transformer.decoder(target_tokens, question_encd, target_mask, question_mask)

            pred_token = output.argmax(2)[:,-1].item()
            output_tokens.append(pred_token)

            if pred_token == end_token:
                break
                
        answer = self.tokenizer.decode(output_tokens)
        
        self.question = question
        self.answer = answer
        self.attention = attention
        self.call_qna = True
        
        return answer, attention

    def plot_attention_weights(self, draw_mean=False):
        if not self.call_qna:
            raise Exception('There is no `question`, `answer` and `attention`. Call `qna` first')
        question_token = to_tokens(self.question, self.tokenizer, to_ids=False)
        question_token = ['<sos>']+question_token+['<eos>']

        answer_token = to_tokens(self.answer, self.tokenizer, to_ids=False)
        answer_token = answer_token+['<eos>']

        attention = self.attention.squeeze(0)
        if draw_mean:
            attention = torch.mean(attention, dim=0, keepdim=True)
        attention = attention.cpu().detach().numpy()

        n_col = 4
        n_row = (attention.shape[0]-1)//n_col + 1
        fig = plt.figure(figsize = (n_col*6, n_row*6))
        for i in range(attention.shape[0]):
            plt.subplot(n_row, n_col, i+1)
            plt.matshow(attention[i], fignum=False)
            plt.xticks(range(len(question_token)), question_token, rotation=45)
            plt.yticks(range(len(answer_token)), answer_token)
        plt.show()
```

`Chatbot`은 `transformer`(모델), `tokenizer`, `device`를 받아 만든다.  
모델과 토크나이저는 모델 훈련시에 저장한 파일들을 불러오면 된다.

#### qna

Transformer의 예측은 일반적인 모델의 예측과는 조금 다르다.  
일반적인 모델의 예측은 input을 넣고 모델에서 output이 나오는 구조이지만 Transformer는 구조에서도 봤듯이 `tar`(output)이 모델의 input으로도 들어가고 loss 계산을 위한 정답으로도 쓰인다.  
따라서 `question`을 넣으면 바로 `answer`가 나오는 구조가 아니다.  

Transformer의 예측을 수행하는 방법은 다음과 같다.

1. input을 Transformer의 `Encoder`에 넣고 self attention의 output(위 코드에서는 `question_encd`)을 계산한다.
2. `Encoder`의 self attention의 output과 `<sos>`토큰을 `Decoder`의 input으로 넣고 `<sos>`토큰의 다음 token으로 나올 단어들의 확률을 얻는다.
3. `<sos>`토큰의 다음 token으로 나올 단어들 중 확률이 가장 높은 단어를 `<sos>`토큰 다음 sequence로 추가하고, 이전에 계산한 `Encoder`의 self attention의 output과 함께 `Decoder`에 input으로 넣는다.
4. 3번을 반복하여 다음 token으로 나올 단어들 중 `<eos>`의 확률이 가장 높을 때 예측을 멈춘다.

#### plot_attention_weights

attention을 시각화하는 method이다. [Attention Visualization](#attention-visualization)에서 다시 설명한다.

### Main

```python
def main(args):

    STOP_SIGN = args.stop_sign
    MODEL_PATH = args.model_path
    MODEL_NAME = args.model_name
    TOKENIZER_PATH = args.tokenizer_path
    TOKENIZER_NAME = args.tokenizer_name

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')

    with open(TOKENIZER_PATH+TOKENIZER_NAME+'.pickle', 'rb') as f:
        tokenizer = pickle.load(f)

    transformer = torch.load(MODEL_PATH+MODEL_NAME+'.pt')

    chatbot = Chatbot(transformer, tokenizer, device)

    print('Start ChatBot\nEnter the message\n')
    print(f'To stop conversation, Enter "{STOP_SIGN}"\n')
    while True:
        question = input('You: ')
        if question == STOP_SIGN:
            break
        answer, _ = chatbot.qna(question)
        print(f'ChatBot: {answer}')


if __name__ == '__main__':

    args = parse_args()
    if args is None:
        exit()

    main(args)
```

멈추기 전까지 `qna`를 반복한다.

## Chatbot을 실행해보자

![chatbot_0](/assets/img/posts/transformer_chatbot-3/chatbot_0.png)

나의 원대한 꿈을 응원해주는 챗봇이 생겼다.

![chatbot_1](/assets/img/posts/transformer_chatbot-3/chatbot_1.png)

해적왕은 응원해주지 않는다.

![chatbot_2](/assets/img/posts/transformer_chatbot-3/chatbot_2.png)

이상한 소리 하면 무시하는 것 같다.

그냥 농담처럼 얘기했지만, 사실 이전 포스팅에서 말했듯 `valid_loss`가 제대로 줄어들지 않았고(물론 지금 사용하는 loss가 완벽하지는 않지만) 훈련이 제대로 되지 않았다고 보는게 맞다.  
train set에 없는 질문에 대한 답변은 그냥 아무말이나 하는 것이라고 봐야한다.

## Attention Visualization

[Chatbot 만들기 (1)]()의 `Transformer`를 보면 `outputs`외에 `attention`도 return하게 되어있다. 이는 각 `query`와 `key`가 어떤 attention을 가지고 있는지 행렬 형태를 return하게 된다.  
이를 이용하여 attention을 시각화 할 수 있다.  

attention의 shape은 `[batch_size, n_heads, query_len, key_len]`로 한개의 데이터에 대해서는 `[n_heads, query_len, key_len]`의 shape을 가지게 된다.  
즉 attention 행렬이 `n_heads`만큼 있는데, 이를 시각화 하는 방법은 두가지이다.

1. 각 head들의 attention을 각각 시각화
2. 모든 head들의 attention을 평균을 취하여 시각화

`draw_mean`을 옵션으로 만들어두었다.

시각화 결과를 살펴보자.

```python
question = '내 꿈은 건물주가 되는거야!'
chatbot = Chatbot(transformer, tokenizer, device)
answer, attention = chatbot.qna(question)

print('Question:', question)
print('Answer:', answer)
chatbot.plot_attention_weights()
```

![attention_visualization](/assets/img/posts/transformer_chatbot-3/attention_visualization.png)

학습이 잘 되었다면 눈으로 보기에도 납득할 수 있는 형태의 attention 행렬이 생긴다. 또한 각 head들의 attention 결과가 크게 다르지 않다.  
보기 좋은 attention 시각화가 나오지 않은 것은 다소 아쉽다.  

## 마무리

Transformer를 만들고, 데이터를 전처리하고, 모델을 학습하고, 챗봇을 만들기까지. 모두다 재미있었다.  
물론 바로 앞에 말한 것 처럼 보기 좋은 attention 행렬이 나오지 않은 것은 속상하다. 번역기같은 Transformer에 좀 더 잘 맞춰진 task를 수행했다면 잘 됐을 것이라 생각한다.  
하지만 챗봇이 더 재미있어보였다. 그리고 실제로 재밌었다.  
챗봇이 대답하는 것도 나름 대화는 되는 느낌이라 만족스러웠다.  

이상으로 챗봇 만들기 포스팅을 마친다.

하지만 **챗봇 만들기**라는 중간 목표가 마무리 되었을 뿐. 아직 **TTS**를 향해 갈길이 멀다.

## 원본 코드 ➞ [<span style="color:#AC1538">CODE (GitHub)</span>](https://github.com/fidabspd/toy/blob/master/chatbot/codes)
