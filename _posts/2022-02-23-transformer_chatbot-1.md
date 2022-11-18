---
title: Transformer - Chatbot 만들기 (1)
excerpt: ""
tags: [Chatbot, NLP, Transformer, PyTorch]
toc: true
toc_sticky : true
use_math: true
---

과거 TTS 프로젝트를 진행해본 적이 있다. 해당 프로젝트 당시 **carpedm20** 님의 [multi-speaker-tacotron-tensorflow](https://github.com/carpedm20/multi-speaker-tacotron-tensorflow)를 참고하여 사용했고, 꽤나 의미있는 결과물을 만들어냈다. 나름의 소스 리뷰를 거쳤고 데이터 수집부터 직접 녹음한 목소리를 함께 train하여 결과물을 만들기도 했지만 아쉬움이 남는다. 그 이유는 '이 프로젝트를 100% 이해하고 있나?'라는 질문에 자신있게 'Yes!'라고 대답하지 못하겠다. 물론 해당 프로젝트에 대한 질문을 받으면 대답하는데 무리는 없다. 하지만 이정도로 만족하지 못하겠다. 그런 의미에서 기존의 아키텍쳐를 답습하여 그대로 만들기보단, 기존의 아키텍쳐를 발전시켜 새로운 TTS를 만들어보자.

## How To Improve

기존에 사용했던 Deep Voice 2 아키텍쳐이다. ([Deep Voice 2: Multi-Speaker Neural Text-to-Speech](https://arxiv.org/abs/1705.08947))

![tacotron](/post_images/transformer_chatbot-1/tacotron.png)

개선하고자 하는 Key는 `seq2seq` ➞ `transformer`이다.

기존에 사용했던 아키텍쳐는 seq2seq에 기반을 둔 어텐션을 사용하고있다.  
그리고 이것을 Transformer 형태로 바꾸고 싶다.  
좀 찾아보니 당연하게도 Transformer를 TTS에 활용한 사례는 많이 보였다. 그중 다음 두 논문을 참고해 TTS를 구현해볼 생각이다.  

- [Neural Speech Synthesis with Transformer Network](https://arxiv.org/abs/1809.08895)  
- [MultiSpeech: Multi-Speaker Text to Speech with Transformer](https://arxiv.org/abs/2006.04664)

그냥 최근 TTS 분야의 SOTA를 파볼까 싶기도 했지만 일단은 한걸음씩 가보자.

## Transformer

순서대로라면 데이터 수집부터 시작하는게 일반적이겠지만 *구조를 이렇게 바꿀거다!*를 선언해버린 만큼 Transformer를 먼저 짚고 넘어가고 싶다.  
어차피 Transformer를 만들어 씌우는게 목적이니, 일단은 Transformer를 이용한 챗봇을 만들어보자. 그 뒤에 데이터 수집부터 시작한다. (솔직히 챗봇 만드는게 재미있을 것 같아서 빨리 하고싶었다..!)

우선은 Transformer를 구현해보자.  

(Attention의 기본 개념은 잘 숙지된 상태를 가정한다. Transformer 또한 개념보다는 코드로 구현하는데에 초점이 있다.)

## 목차

1. [**Chatbot 만들기 (1)**](https://fidabspd.github.io/2022/02/23/transformer_chatbot-1.html)
1. [Chatbot 만들기 (2)](https://fidabspd.github.io/2022/03/01/transformer_chatbot-2.html)
1. [Chatbot 만들기 (3)](https://fidabspd.github.io/2022/03/02/transformer_chatbot-3.html)

## 원본 코드 ➞ [<span style="color:#AC1538">CODE (GitHub)</span>](https://github.com/fidabspd/toy/blob/master/chatbot/codes/transformer_torch.py)

### Architecture

![transformer_architecture](/post_images/transformer_chatbot-1/transformer_architecture.png)

그 유명한 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)에 소개된 아키텍쳐이다.

이를 최대한 간단하게 설명하면 다음과 같다.

- `Inputs`와 `Outputs`는 `positional_encoding`을 수행한다.
    - `Outputs`는 `shifted right`하여 들어간다.
- `Encoder`와 `Decoder`로 이루어져있다.
- `Encoder`는
    - `self_attention`
    - `feed_forward`  
    로 이루어져있다.
- `Decoder`는
    - `self_attention`
    - `Encoder`의 output을 K(Key), V(Value)로 사용하고, `Decoder self_attention`의 output을 Q(Query)로 사용한 `Encoder-Decoder attention`  
    (Q, K, V는 뒤에서 마저 설명)
    - `feed_forward`  
    로 이루어져있다.
- 각 `attention`과 `feed_forward`가 끝날 때마다 `layer_normalization`과 `residual_connection`을 수행한다.
- 각 `Encoder`와 `Decoder`는 `N`번 반복하여 stack을 쌓는다.
- `Decoder`의 최종 output에 `Linear`와 `Softmax`를 붙인다.

각 `attention`은 `Multi-Head Attention`이라고 부르며 구조는 다음과 같다.

![multi_head_attention](/post_images/transformer_chatbot-1/multi_head_attention.png)

- `Multi-Head Attention`은 `Scaled Dot-Product Attention`이 $h(=number\, of\, heads)$번 반복된 구조이다.
    - $h$는 $d_{model}$의 약수여야한다.
- `Scaled Dot-Product Attention`은 `Linear`를 지난 `Q`, `K`, `V`를 이용한다.  
(`Scaled Dot-Product Attention`에 대한 자세한 설명은 생략)
- `Scaled Dot-Product Attention`의 결과들을 `Concat`하여 `Linear`를 이용하여 최종 output을 만든다.

`Scaled Dot-Product Attention`의 계산은 한줄의 식으로 표현이 가능하다.  

$Attention(Q, K, V) = softmax\bigg(\dfrac{QK^T}{\sqrt{d_k}}\bigg)V\;, \quad \mathbf{where} \; d_k = \dfrac{d_{model}}{h(=number\, of\, heads)}$  
(자세하게 설명하자면 복잡하지만, $d_{model}$은 `Embedding`의 output차원이라고 생각하면 편하다.)

`masking`에 대한 설명이 조금은 부족하지만, 코드부분에서 추가적인 설명으로 다루도록한다.  
우선 여기까지만 알면 **Transformer**를 만들 수 있다.  

PyTroch를 이용하여 Transformer를 만들어보자.

**PyTorch를 택한 이유**  
사실 Tensorflow의 subclassing 형태로 만들고있다가 ([텐플 코드](https://github.com/fidabspd/ml_basic/blob/master/transformer/codes/transformer_tf.py)) PyTroch로 돌렸다.  
여러가지 이유가 있었는데 가장 결정적이었던 이유는 텐서보드 그래프가 너무 예쁘지 않았다. 잘 몰라서 못그리는건지, 아니면 어쩔 수 없는건지 모르겠지만 텐서플로우는 텐서보드 그래프가 파이토치에 비해 가시성이 좋지 않았다.  
사소한 이유로는 `tf.keras.layers.Layer`와 `tf.keras.models.Model`과 `tf.Module`의 경계가 모호하게 느껴졌다. 어차피 subclassing형태로 epoch과 batch를 직접 만들 생각이었기에 keras의 `fit`이나 `predict`, `Callback`등을 사용할 일은 없었다. 그래서 세개가 더더욱 애매하게 느껴졌다.

반면에 파이토치는 텐서보드 그래프 가시성도 좋았고 `nn.Module`도 직관적으로 느껴졌다.

## CODE

다음 자료들을 참고하여 만들었다.  
[tensorflow.org](https://www.tensorflow.org/text/tutorials/transformer)  
[딥러닝을 이용한 자연어 처리 입문](https://wikidocs.net/31379)  
[ndb796/Deep-Learning-Paper-Review-and-Practice](https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/Attention_is_All_You_Need_Tutorial_(German_English).ipynb)

\+ 좋은 자료 올려주시는 분들 정말 감사합니다.

### Libraries

```python
import torch
from torch import nn
```

### Multi-Head Attention

```python
class MultiHeadAttentionLayer(nn.Module):
    
    def __init__(self, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()

        assert hidden_dim % n_heads == 0, f'hidden_dim must be multiple of n_heads.'
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim//n_heads
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

        # in_shape: [batch_size, seq_len, hidden_dim]
        # out_shape: [batch_size, seq_len, hidden_dim]
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)

        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

    def split_heads(self, inputs, batch_size):
        inputs = inputs.view(batch_size, -1, self.n_heads, self.head_dim)
        # [batch_size, seq_len, n_heads, head_dim]
        splits = inputs.permute(0, 2, 1, 3)
        return splits  # [batch_size, n_heads, seq_len, head_dim] -> n_heads를 앞으로

    def scaled_dot_product_attention(self, query, key, value, mask):
        key_t = key.permute(0, 1, 3, 2)
        energy = torch.matmul(query, key_t) / self.scale  # [batch_size, n_heads, query_len, key_len]
        # mask shape:
        # for inp self_attention: [batch_size, 1, 1, key_len(inp)]
        # for tar self_attention: [batch_size, 1, query_len(tar)(=key_len(tar)), key_len(tar)(=query_len(tar))]
        # for encd_attention: [batch_size, 1, 1, key_len(inp)]
        if mask is not None:
            energy = energy.masked_fill(mask==0, -1e10)  # key에 masking
            # `masked_fill`의 parameter로 받는 `mask==0`에 대해. 
            # - `energy`와 shape의 차원의 개수가 달라도 괜찮다. `energy`와 `mask==0`의 차원 개수중 더 많은 차원의 개수를 가지도록 자동으로 맞춘다.
            # - 각 차원의 len은 `energy`의 각 차원 len과 일치하거나 1이어야한다. (배수는 안된다.)
        attention = torch.softmax(energy, axis=-1)  # axis=-1 은 key의 문장 위치
        attention = self.dropout(attention)
        # attention shape: [batch_size, n_heads, query_len, key_len(=value_len)]
        # value shape: [batch_size, n_heads, value_len(=key_len), head_dim]
        x = torch.matmul(attention, value)  # [batch_size, n_heads, query_len, head_dim]
        return x, attention

    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        x, attention = self.scaled_dot_product_attention(query, key, value, mask)
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, query_len, n_heads, head_dim]
        x = x.view(batch_size, -1, self.hidden_dim)  # [batch_size, query_len, hidden_dim]

        outputs = self.fc_o(x)
        
        return outputs, attention  # [batch_size, query_len, hidden_dim], [batch_size, n_heads, query_len, key_len(=value_len)]
```

분명 Multi-Head Attention 아키텍쳐 그림에는 `V`, `K`, `Q`가 head의 개수만큼 따로따로 input되는 것 처럼 보였지만 코드를 보면 그렇지 않다.  
행렬 곱을 통한 일종의 트릭을 이용한 셈인데, `self.fc_q`, `self.fc_k`, `self.fc_v`를 지나기 전의 `Q`, `K`, `V`는 각 row number에 해당하는 seq의 token embedding결과가 나열되어있는 `[batch_size, seq_len, hidden_dim(=d_model=n_heads*head_dim)]` shape을 가진 형태이다.  
이를 column별로 split하게 되면 embedding 벡터를 쪼개는 것이니 embedding 결과가 없어지는 것이나 마찬가지지만 `self.fc_q`, `self.fc_k`, `self.fc_v`를 지난 뒤에는 split해도 괜찮은 형태가 된다.  
따라서 이를 $h$개로 split해줌으로써 `[batch_size, seq_len, n_heads, head_dim]` shape으로 변형한다. 방법은 `view`(=`reshape`과 비슷한 역할 (`contiguous`에 대한 설명은 생략))를 이용한다.  
그리고 `permute`(=`transpose`)를 이용하여 `[batch_size, n_heads, seq_len, head_dim]` shape으로 바꾸면 n_heads는 분리한채로 연산이 가능하다.  
`scaled_dot_product_attention`의 연산이 끝나면 다시 `[batch_size, seq_len, n_heads, head_dim]` shape으로 바꾸고, 한번 더 `[batch_size, seq_len, hidden_dim]` shape으로 바꾼 뒤 `self.fc_o`를 통해 내보낸다.

mask의 구조에 대해서는 [Transformer](#transformer)에서 좀 더 자세히 말하기로 한다. 간단히만 말하자면 `key`에 대해 참고하면 안되는 값들을 가려주는 장치이다.

추가적으로 `energy`를 $\sqrt{d_k}$로 나누어 scaling을 수행한다.  

이에 관해서는 하고싶은 말이 상당히 많기 때문에 별도의 포스팅을 따로 작성해보겠다. - [**Transformer의 scaling에 대한 고찰**](https://fidabspd.github.io/2022/03/02/analysis_for_transformer.html)

### Position-Wise Fully Connected Feed-Forward

```python
class PositionwiseFeedforwardLayer(nn.Module):

    def __init__(self, pf_dim, hidden_dim, dropout_ratio):
        super().__init__()
        self.fc_0 = nn.Linear(hidden_dim, pf_dim)
        self.fc_1 = nn.Linear(pf_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, inputs):
        x = torch.relu(self.fc_0(inputs))
        x = self.dropout(x)
        outputs = self.fc_1(x)
        return outputs
```

큰 특징 없다. 두번의 `Linear`, 한번의 `Dropout`, 한번의 `relu`.  
두번의 `Linear`를 지나며 거치게 되는 중간 dimension인 `pf_dim`의 경우에는 논문에서는 $d_{ff} = 2048$을 제시한다. (`hidden_dim`은 $d_{model}=512$)

### Encoder Layer

```python
class EncoderLayer(nn.Module):

    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout_ratio = dropout_ratio

        self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.pos_feedforward = PositionwiseFeedforwardLayer(pf_dim, hidden_dim, dropout_ratio)
        self.pos_ff_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, inputs, mask=None):
        attn_outputs, _ = self.self_attention(inputs, inputs, inputs, mask)
        attn_outputs = self.dropout(attn_outputs)
        attn_outputs = self.self_attn_norm(inputs+attn_outputs)  # residual connection

        ff_outputs = self.pos_feedforward(attn_outputs)
        ff_outputs = self.dropout(ff_outputs)
        ff_outputs = self.pos_ff_norm(attn_outputs+ff_outputs)  # residual connection

        return ff_outputs  # [batch_size, query_len(inp), hidden_dim]
```

[Architecture](#architecture)에서 설명한 구조에서 `Dropout`을 몇번 추가한 것 말고는 완벽히 같다.  
눈여겨 볼 점은 self attention인 만큼 `query`, `key`, `value`에 `input`으로 모두 같은 값이 들어간다는 점이다.

### Decoder Layer

```python
class DecoderLayer(nn.Module):

    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
        super().__init__()
        
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.encd_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
        self.encd_attn_norm = nn.LayerNorm(hidden_dim)
        self.pos_feedforward = PositionwiseFeedforwardLayer(pf_dim, hidden_dim, dropout_ratio)
        self.pos_ff_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, target, encd, target_mask, encd_mask):
        self_attn_outputs, _ = self.self_attention(target, target, target, target_mask)
        self_attn_outputs = self.dropout(self_attn_outputs)
        self_attn_outputs = self.self_attn_norm(target+self_attn_outputs)
        
        # self_attn_outputs shape: [batch_size, query_len(tar), hidden_dim]
        # encd shape: [batch_size, query_len(inp), hidden_dim]
        # new_query_len = query_len(tar); new_key_len(=new_val_len) = query_len(inp)
        encd_attn_outputs, attention = self.encd_attention(self_attn_outputs, encd, encd, encd_mask)
        encd_attn_outputs = self.dropout(encd_attn_outputs)
        encd_attn_outputs = self.encd_attn_norm(self_attn_outputs+encd_attn_outputs)

        outputs = self.pos_feedforward(encd_attn_outputs)
        outputs = self.dropout(outputs)
        outputs = self.pos_ff_norm(encd_attn_outputs+outputs)

        return outputs, attention  # [batch_size, query_len(tar), hidden_dim]
```

마찬가지로 `Dropout`을 몇번 추가한것 말고는 [Architecture](#architecture)에서 설명한 구조와 완벽히 같다.  

눈 여겨볼 점은 다음 두가지.

1. 첫 `attention`은 `self attention`으로, `query`, `key`, `value` 모두 `target`으로 같다.  
2. 두번째 `attention`은 `Encoder-Decoder attention`으로, `target`이 `query`로 들어가고,  
`Encoder`의 output인 `encd`가 `key`와 `value`로 들어간다.

### Encoder Stacks

```python
class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, n_layers, n_heads, pf_dim,
                 dropout_ratio, device, max_seq_len=100):
        # input_dim = encoder vocab_size
        super().__init__()
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

        self.tok_emb = nn.Embedding(input_dim, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.encd_stk = nn.ModuleList([
            EncoderLayer(hidden_dim, n_heads, pf_dim, dropout_ratio, device)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        emb = self.tok_emb(x) * self.scale + self.pos_emb(pos)
        outputs = self.dropout(emb)

        for layer in self.encd_stk:
            outputs = layer(outputs, mask)

        return outputs
```

[Architecture](#architecture)에서 설명한대로 [Encoder Layer](#encoder-layer)를 n번 쌓는 과정이다.

`positional encoding`은 `positional embedding`으로 대체한다.  
이는 **BERT**등의 Transformer 기반으로 발전한 아키텍쳐에서 쓰이는 방법이기도 하다.  
각 sequence number를 embedding 레이어에 넣어주면 된다.

각 token의 embedding에 대해서는 $\sqrt{d_{model}}$을 곱해준다.  

이 역시도 [Multi-Head Attention](#multi-head-attention)의 `energy` scaling과 더불어 별도의 포스팅에서 다뤄보겠다. - [**Transformer의 scaling에 대한 고찰**](https://fidabspd.github.io/2022/03/02/analysis_for_transformer.html)

### Decoder Stacks

```python
class Decoder(nn.Module):
    
    def __init__(self, output_dim, hidden_dim, n_layers, n_heads, pf_dim,
                 dropout_ratio, device, max_seq_len=100):
        # output_dim = decoder vocab_size
        super().__init__()
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

        self.tok_emb = nn.Embedding(output_dim, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.decd_stk = nn.ModuleList([
            DecoderLayer(hidden_dim, n_heads, pf_dim, dropout_ratio, device)
            for _ in range(n_layers)
        ])

        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, target, encd, target_mask, encd_mask):
        batch_size = target.shape[0]
        seq_len = target.shape[1]

        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        emb = self.tok_emb(target) * self.scale + self.pos_emb(pos)
        outputs = self.dropout(emb)

        for layer in self.decd_stk:
            outputs, attention = layer(outputs, encd, target_mask, encd_mask)

        outputs = self.fc_out(outputs)  # [batch_size, query_len(tar), decoder vocab_size]

        return outputs, attention
```

역시나 [Architecture](#architecture)에서 설명한대로 [Decoder Layer](#decoder-layer)를 n번 쌓는 과정이다.  
[Encoder Stacks](#encoder-stacks)와 마찬가지로 token embedding에 $\sqrt{d_{model}}$을 곱하고 positional embedding을 더한다.  

`Encoder`와 크게 다른점 한가지는 최종 output에 `Linear`가 하나 붙어있다는 것이다.  
아키텍쳐 상으로는 `Linear`뒤에 `softmax`가 하나 더 붙어있지만 `CrossEntropyLoss`를 쓸 것이기 때문에 `softmax`는 생략한다.  
output shape은 `[batch_size, query_len(tar), decoder vocab_size]`로, 현재 seq의 다음에 등장할 decoder vocab의 확률을 output으로 반환한다.

### Transformer

```python
class Transformer(nn.Module):

    def __init__(self, input_dim, output_dim, n_layers, hidden_dim, n_heads, pf_dim,
                 in_seq_len, out_seq_len, pad_idx, dropout_ratio, device):
        super().__init__()
        self.device = device

        self.out_seq_len = out_seq_len

        self.encoder = Encoder(
            input_dim, hidden_dim, n_layers, n_heads, pf_dim,
            dropout_ratio, device, in_seq_len
        )
        self.decoder = Decoder(
            output_dim, hidden_dim, n_layers, n_heads, pf_dim,
            dropout_ratio, device, out_seq_len
        )
        self.pad_idx = pad_idx

    def create_padding_mask(self, key, for_target=False):
        mask = (key != self.pad_idx).unsqueeze(1).unsqueeze(2)
        if for_target:
            key_len = key.shape[1]
            target_sub_mask = torch.tril(torch.ones((key_len, key_len), device = self.device)).bool()
            mask = mask & target_sub_mask
        return mask  # [batch_size, 1, 1, key_len]

    def forward(self, inp, tar):
        inp_mask = self.create_padding_mask(inp)
        tar_mask = self.create_padding_mask(tar, True)

        enc_inp = self.encoder(inp, inp_mask)
        output, attention = self.decoder(tar, enc_inp, tar_mask, inp_mask)
        # output shape: [batch_size, query_len(tar), decoder vocab_size]
        # attention_shape: [batch_size, n_heads, query_len(tar), key_len(inp)]

        return output, attention
```

마스크를 생성하는 부분 말고는 뻔한 내용이다. 위에서 만든 [Encoder Stacks](#encoder-stacks)와 [Decoder Stacks](#decoder-stacks)를 합해 최종으로 `Transformer`를 만든다.

마스크는 두가지를 가린다.

1. `key`의 패딩 토큰 `<pad>`를 가린다.
2. target이 self attention을 수행할 때 현재 seq보다 뒤쪽 seq를 가린다.

Transformer의 예측은 한 seq뒤를 예측하게 되는데 self attention에서 target의 현재 seq보다 뒤에 있느 값들을 보여주게 되면 사실상 정답을 input으로 넣어주는 것이나 다름이 없다. 따라서 이를 참고하지 못하도록 가려줘야한다.  
mask로 가려줘야하는 부분은 [Multi-Head Attention](#multi-head-attention)에서 해당 부분의 key에 해당하는 `energy`를 아주 작은 음수값으로 만들어 `softmax`를 지나면서 0에 아주 가까운 값으로 만든다.

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)에 소개된 Transformer 구조를 PyTorch를 이용하여 구현해보았다.  
분량이 길어져 이를 훈련하고, Chatbot을 만들고 실행하는 내용은 다음 포스팅에서 계속하도록 하자.

## 목차

1. [**Chatbot 만들기 (1)**](https://fidabspd.github.io/2022/02/23/transformer_chatbot-1.html)
1. [Chatbot 만들기 (2)](https://fidabspd.github.io/2022/03/01/transformer_chatbot-2.html)
1. [Chatbot 만들기 (3)](https://fidabspd.github.io/2022/03/02/transformer_chatbot-3.html)

## 원본 코드 ➞ [<span style="color:#AC1538">CODE (GitHub)</span>](https://github.com/fidabspd/toy/blob/master/chatbot/codes/transformer_torch.py)
