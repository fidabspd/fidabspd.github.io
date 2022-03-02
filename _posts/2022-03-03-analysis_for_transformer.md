---
layout: post
title: Transformer의 Scaling에 대한 고찰
tags: [Transformer, PyTorch]
excerpt_separator: <!--more-->
use_math: true
---

[Chatbot 만들기 (1)](https://fidabspd.github.io/2022/02/23/transformer_chatbot-1.html)의 [Multi-Head Attention](https://fidabspd.github.io/2022/02/23/transformer_chatbot-1.html#multi-head-attention)과 [Encoder Stacks](https://fidabspd.github.io/2022/02/23/transformer_chatbot-1.html#encoder-stacks), [Decoder Stacks](https://fidabspd.github.io/2022/02/23/transformer_chatbot-1.html#decoder-stacks)에 등장하는 scaling에 대해 다른 포스팅에서 따로 다루겠다고 했다.  
그에 대한 내용이다.

앞서 말한것 처럼 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)에서 제시하는 Transformer 아키텍쳐에는 두번의 scaling이 있다.  
논문에 등장하는 순서대로 보면 다음과 같다.  <!--more-->

1. Multi-Head Attention의 Query와 Key의 행렬곱(통칭 **energy**)을 $\sqrt{d_k}$로 나눈다.  
>$Attention(Q, K, V) = softmax\bigg(\dfrac{QK^T}{\sqrt{d_k}}\bigg)V$

1. Input과 Target의 embedding 벡터에 $\sqrt{d_{model}}$를 곱한다.
>In the embedding layers, we multiply those weights by $d_{model}$.

해당 순서대로 살펴보도록 하자.

## Scaled Dot-Product Attention

Multi-Head Attention은 $h(=number\,of\,heads)$개의 Scaled Dot-Product Attention으로 이루어져있다.  
그리고 Scaled Dot-Product Attention은 그 이름에 걸맞게 Scaling을 한다.

그럼 왜 scaling을 하는지. 우선 논문을 읽어보자.

>The two most commonly used attention functions are additive attention, and dot-product (multi-plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of $\frac{1}{\sqrt{d_k}}$ . Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.
>
>While for small values of d k the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of $\sqrt{d_k}$ . We suspect that for large values of $\sqrt{d_k}$ , the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $\frac{1}{\sqrt{d_k}}$ .

간단하게 의역을 보태 해석해보자면

>보통 쓰는 attention은 additive attention과 dot-product attention이 있는데, 우리는 dot-product attention을 썼다.  
>그리고 예상컨데 dot-product attention은 $\sqrt{d_k}$가 큰 값을 가질 때, 내적값이 매우 커져 softmax function에서 gradient가 소실 되는 것 같다.

솔직하게 처음 읽었을 때 들은 생각은  
'**suspect...?** 확신이 없었나?'

어찌됐건 명확한 답을 내려주진 않는다. 주체적으로 생각해보자.

**<span style="color:#AC1538">여기서부터는 모두 개인적인 생각이니 유의.</span>**

두가지 측면에서 생각해보자.

1. 왜 scaling을 하는지.
2. 그 값이 왜 하필 $\dfrac{1}{\sqrt{d_k}}$인지.

### scaling이 필요한 이유

왜 scaling을 해야하는지. 사실 그 이유는 간단하다고 생각한다. 논문의 말도 있고 softmax의 특징을 수학적으로 생각해보면 scaling은 당연히 필요하다.  
`softmax([1, 2, 3])`과 `softmax([1+1e10, 2+1e10, 3+1e10])`의 값은 같다. (이유는 softmax 공식을 써보면 간단하니 모르겠다면 공식을 손으로 써보길 추천.) 원하는바가 이것이 아님은 분명하다.  
이 말은 곧 `softmax([1, 1+1e10, 1])`과 `softmax([1-1e10, 1, 1-1e10])`의 값도 같다. 즉 유독 큰 한두개의 값이 있다면 나머지 값이 얼마든 그 값은 없는 것이나 마찬가지다. softmax는 그 값이 0이나 1에 가까워질 수록 미분값도 0에 가까워지므로 큰값이 하나만 있다면 gradient vanishing 관점에서 최악이다.

그리고 조금 더 생각해보면 유독 큰 한 값이 생기는 것도 당연하다. 그 이유는 self attention 때문이다.  
물론 Linear를 한번 거친 뒤의 값들이기는 하지만 Query와 Key가 같은 경우에 값이 커질 것은 뻔하다.

scaling을 해줘야하는 이유에 대해서는 알았다.  

### 그럼 왜 하필 $\frac{1}{\sqrt{d_k}}$ 일까

**너무 큰 값을 줄이기 위해 나눈다**는 컨셉은 확실하게 알았으므로 **얼마나 커질까?** 를 생각해보면 되겠다.  
dot product에서 가장 값이 커지는 경우는 완벽히 같은 벡터의 내적을 수행할 때이다. 그리고 이 값은 벡터 길이의 제곱으로 나타나게 된다.  

그럼 벡터의 길이는 어떻게 될까?  

이걸 알려면 weight initialize가 중요한데 우선은 표준 정규분포로 가정해보자.  
그럼 벡터의 각 원소들을 확률변수 $X$라고 한다면 $X \sim N(0, 1)$ 이다.  
해당 벡터들의 L2 Norm 값을 새로운 확률변수 $C$라고 한다면 $C$는 자유도가 벡터의 길이인 카이제곱분포를 따른다. 즉 $C \sim \chi(d_k)$  
그리고 카이제곱분포의 평균은 자유도와 같다.  

다른 말로하면, 표준 정규분포로 벡터를 initialize했을 때 해당 벡터들 길이의 기대값은 $\sqrt{d_k}$이다.  

다시 본 문제로 돌아와서, $QK^T$의 각 원소들은 Query와 Key의 벡터간 내적값들의 집합이고 각각의 값들은 완벽히 같은 벡터의 내적일 때 최대가 된다. 그리고 그 최대값의 기대값은 $\sqrt{d_k}$이다.  

수학적으로 파느라 시간이 좀 오래걸렸지만 결국 컨셉 자체는 **최대 길이를 가지는 벡터의 길이 기대값으로 나누어 scaling한다.** 가 컨셉이다.

### 그렇다면 $\frac{1}{\sqrt{d_k}}$보다 좋은 값은 없을까

사실 $\frac{1}{\sqrt{d_k}}$ 이 scaling이 **완벽한가**에 대해서는 잘 모르겠다. 수학적으로 아름다워보이지는 않는다. (지식이 짧아서 그럴수도 있다.)  
weight initialize에 따라서도 미치는 영향이 달라질 것으로도 생각되고, 뒤에서 다루게 될 embedding 벡터에 $\sqrt{d_{model}}$를 곱해 scaling하는 과정이 있는데 $\sqrt{d_k}$가 벡터 길이의 최대값의 기대값이 맞긴 할까?  

그리고 한가지 더 큰 의문점. 결국 $QK^T$ 각 원소의 최대값의 기대값은 벡터 길이가 아니라 L2 Norm인데 **길이의 제곱으로 나눠도 되지 않나?**  
궁금한건 못참는다. 이건 실험해보기도 쉽다. 해보자.  

#### 실험

```python
parser.add_argument('--que_max_seq_len', type=int, default=50)
parser.add_argument('--ans_max_seq_len', type=int, default=50)
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--pf_dim', type=int, default=512)
parser.add_argument('--dropout_ratio', type=float, default=0.1)

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=5e-4)
parser.add_argument('--clip', type=int, default=1)
parser.add_argument('--n_epochs', type=int, default=50)
```

나머지 여러 파라미터들은 다음과 같이 설정했다.  
이를 유지한채로 기존의 방법대로 $\sqrt{d_k}$로 나눴을 때와, 새로운 방법 $d_k$로 나눴을 때를 비교해보자.

데이터는 [Chatbot 만들기 (1)](https://fidabspd.github.io/2022/02/23/transformer_chatbot-1.html)에서 사용했던 데이터를 그대로 사용했다.  

#### $\sqrt{d_k}$

```
------------------------------
Epoch: 01
loss:  48.095781  [ 2368/11823]
loss:  43.838255  [ 4736/11823]
loss:  41.857821  [ 7104/11823]
loss:  40.555370  [ 9472/11823]
loss:  39.521442  [11823/11823]
Train Loss: 39.521
Epoch Time: 0m 9s

...

------------------------------
Epoch: 50
loss:   0.279154  [ 2368/11823]
loss:   0.304162  [ 4736/11823]
loss:   0.311693  [ 7104/11823]
loss:   0.318479  [ 9472/11823]
loss:   0.325430  [11823/11823]
Train Loss: 0.325
Epoch Time: 0m 9s
```

epoch 50 기준 Train Loss가 0.325까지 줄어들었다. 

#### $d_k$

```
------------------------------
Epoch: 01
loss:  47.284764  [ 2368/11823]
loss:  43.755399  [ 4736/11823]
loss:  41.589943  [ 7104/11823]
loss:  40.369123  [ 9472/11823]
loss:  39.297000  [11823/11823]
Train Loss: 39.297
Epoch Time: 0m 9s

...

------------------------------
Epoch: 50
loss:   0.190272  [ 2368/11823]
loss:   0.230853  [ 4736/11823]
loss:   0.243229  [ 7104/11823]
loss:   0.255112  [ 9472/11823]
loss:   0.263232  [11823/11823]
Train Loss: 0.263
Epoch Time: 0m 9s
```

epoch 50 기준 Train Loss가 0.263까지 줄어들었다.  

**어라..?**  
단순한 우연인 걸까. 실험을 각각 5번 반복해봤다.  

$\sqrt{d_k}$를 사용했을 때는 0.31 아래로 단 한번도 내려가지 못한 반면  
$d_k$를 사용했을 때는 0.28 위로 올라간적이 한번도 없다.

또한 loss 수렴 속도도 $d_k$를 사용했을 때가 더 빠르다.

현재 사용하는 데이터와 파라미터 상으로는 $d_k$로 나누는 것이 좋다고 생각해도 되겠다.

#### 이유가 뭘까

사실 확신이 서진 않지만 그래도 끄적여보자면,  
softmax의 미분값은 해당 값이 0.5일때 최대값을 가진다. 그래서 gradient vanishing을 최대한 줄이려면 0 이나 1 가까이보다는 0.5 근처에 모여있는게 좋다.  
하지만 softmax는 모든 합이 1이므로 $d_k$의 값이 커질수록 각각의 값은 필연적으로 0에 가까워질 수 밖에 없다.  

그래서 각각을 softmax에 넣기 전 각각의 값들을 적당히 평탄하게 만들어주는게 좋다. 그렇다고 너무 평평해지면 미분값이 커야할 값들이 별다른 차별점을 가지지 못하게 되는 수도 있으니 scaling에는 적당한 값이 필요할 것이다. 그 적절한 값이 정확히 어디인지는 모르겠지만 현재 상황에서는 $\sqrt{d_k}$ 보다는 $d_k$에 가깝지 않았나 하는 생각이다.

번외로 $\sqrt{d_k}$ 도 $d_k$ 도 아닌 훨씬 더 큰값인 $d_k^2$ 으로 나누어 scaling 해주는 실험도 몇번 반복 해봤다.  
결론부터 말하면 현재 상황에서 $\sqrt{d_k}$ 보다는 성능이 좋았고 $d_k$ 보다는 근소하게 비슷하거나 좋지 않았다.  

최적의 scaling 값이 있을텐데... 궁금하다...!

**softmax 미분 참고**  
$\mathbf{Let} \; a = \dfrac{e^{z_1}}{e^{z_1}+e^{z_2}+e^{z_3}}\;, \quad \dfrac{\partial a}{\partial z_1} = a(1-a)$

Scaled Dot-Product Attention의 scaling에 대해서는 여기서 마무리하고 Input과 Target의 embedding 벡터에 $\sqrt{d_{model}}$를 곱하는 부분으로 넘어가보자.

## Embedding Vector Scaling

>The reason we increase the embedding values before addition is to make the positional encoding relatively smaller. This means the original meaning in the embedding vector won’t be lost when we add them together.
