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

1. Multi-Head Attention의 Query와 Key의 행렬곱을 $\sqrt{d_k}$로 나눈다.  
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
**suspect...?** 갓 구글님께서는 확신이 없으셨나보다.

논문은 명확한 답을 제시해주지 않았으니 주체적으로 생각해보자.

**<span style="color:#AC1538">여기서부터는 모두 개인적인 의견과 뇌피셜 범벅이니 유의.</span>**

두가지 측면에서 생각해보자.

1. 왜 scaling을 하는지.
2. 그 값이 왜 하필 $\frac{1}{\sqrt{d_k}}$인지.

### scaling이 필요한 이유

왜 scaling을 해야하는지. 사실 그 이유는 간단하다고 생각한다. 논문의 말도 있고 softmax의 특징을 수학적으로 생각해보면 scaling은 당연히 필요하다.  
`softmax([1, 2, 3])`과 `softmax([1+1e10, 2+1e10, 3+1e10])`의 값은 같다. 원하는바가 이것이 아님은 분명하다.  
이 말은 곧 `softmax([1, 1+1e10, 1])`과 `softmax([1-1e10, 1, 1-1e10])`의 값도 같다. 즉 유독 큰 한두개의 값이 있다면 나머지 값들은 작은 정도가 얼마든 상관없이 softmax를 지나면 사실상 0이나 다름없어진다. 또한 softmax는 그 값이 0이나 1에 가까워질 수록 미분값도 0에 가까워지므로 큰값이 딱 하나만 있다면 gradient vanishing 관점에서는 더더욱 최악이다.

그리고 조금 더 생각해보면 유독 큰 한 값이 생기는 것도 당연하다. 그 이유는 self attention 때문이다.  
물론 Linear를 한번 거친 뒤의 값들이기는 하지만 Query와 Key가 같은 경우에 값이 커질 것은 뻔하다.

scaling을 해줘야하는 이유에 대해서는 알았다.  

### 그럼 왜 하필 $\frac{1}{\sqrt{d_k}}$ 일까

**너무 큰 값을 줄이기 위해 나눈다**는 컨셉은 확실하게 알았으므로 **얼마나 커질까?** 를 생각해보면 되겠다.  
dot product에서 가장 값이 커지는 경우는 완벽히 같은 벡터의 내적을 수행할 때(즉 L2 Norm을 계산하게 될 때)이다.  

그럼 벡터의 길이는 어떻게 될까? (본 경우에 벡터의 차원은 $d_k$이다.)  

이걸 알려면 weight initialize가 중요한데 우선은 표준 정규분포로 가정해보자.  

$X \sim N(0, 1)\;, \quad \mathbf{where} \; X\,is\,element\,of\,\vec{V} \in \mathbb{R}^{d_k}$  

이 때 벡터 $\vec{V}$의 L2 Norm을 새로운 확률변수 $C$라고 한다면 $C$는 자유도가 $\vec{V}$의 차원인 카이제곱분포를 따른다.  

$C \sim \chi(d_k)\;, \quad \mathbf{where} \; C = \left\| \left\| \vec{V} \right\| \right\|$

(해당 계산의 유도는 상당히 복잡하므로 생략.)  

그리고 카이제곱분포의 평균은 자유도와 같다.

$\overline{C} = d_k$

다른 말로하면, 표준 정규분포로 어떤 벡터를 initialize했을 때 그 벡터의 L2 Norm 기대값은 해당 벡터의 차원과 같다. (본 케이스에서는 $d_k$)  

다시 본 문제로 돌아와서, $QK^T$행렬의 각 원소들은 Query와 Key의 벡터간 내적값들의 집합이고 각각의 값들은 완벽히 같은 벡터의 내적일 때 (즉, 한 벡터의 L2 Norm을 계산하게 될 때) 최대가 된다. 그리고 그 최대값의 기대값은 $d_k$이다.  

수학적으로 설명하느라 돌아왔지만 결국 컨셉 자체는 **각 Head에 들어오는 Query와 Key 벡터의 길이 기대값으로 나누어 scaling한다.** 가 컨셉이다.

### 그렇다면 $\frac{1}{\sqrt{d_k}}$보다 좋은 값은 없을까

사실 $\frac{1}{\sqrt{d_k}}$ 이 scaling이 **완벽한가**에 대해서는 잘 모르겠다. 수학적으로 아름다워보이지는 않는다. (지식이 짧아서 그럴수도 있다.)  
weight initialize에 따라서도 미치는 영향이 달라질 것으로도 생각되고, 뒤에서 다루게 될 embedding 벡터에 $\sqrt{d_{model}}$를 곱해 scaling하는 과정이 있는데 $\sqrt{d_k}$가 벡터 길이의 기대값이 맞긴 할까? 개인적으로는 아니라고 생각한다.  

그리고 한가지 더 큰 의문점. 결국 $QK^T$ 각 원소의 최대값의 기대값은 벡터 길이가 아니라 L2 Norm인데 길이로 나눌 것이 아니라 길이의 제곱인 **L2 Norm으로 나눠도 되지 않나?**  
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

데이터는 [Chatbot 만들기 (2)](https://fidabspd.github.io/2022/03/01/transformer_chatbot-2.html)에서 사용했던 데이터를 그대로 사용했다.  

#### 원래 방식 - $\sqrt{d_k}$ 로 나누기

```
------------------------------
Epoch: 01
Train Loss: 39.521
Epoch Time: 0m 9s
------------------------------
Epoch: 02
Train Loss: 32.778
Epoch Time: 0m 9s
------------------------------
Epoch: 03
Train Loss: 28.304
Epoch Time: 0m 9s

...

------------------------------
Epoch: 48
Train Loss: 0.349
Epoch Time: 0m 9s
------------------------------
Epoch: 49
Train Loss: 0.339
Epoch Time: 0m 9s
------------------------------
Epoch: 50
Train Loss: 0.325
Epoch Time: 0m 9s
```

epoch 50 기준 Train Loss가 0.325까지 줄어들었다. 

#### 실험 방식 - $d_k$ 로 나누기

```
------------------------------
Epoch: 01
Train Loss: 39.297
Epoch Time: 0m 9s
------------------------------
Epoch: 02
Train Loss: 31.671
Epoch Time: 0m 9s
------------------------------
Epoch: 03
Train Loss: 26.675
Epoch Time: 0m 9s

...

------------------------------
Epoch: 48
Train Loss: 0.271
Epoch Time: 0m 9s
------------------------------
Epoch: 49
Train Loss: 0.266
Epoch Time: 0m 9s
------------------------------
Epoch: 50
Train Loss: 0.263
Epoch Time: 0m 9s
```

epoch 50 기준 Train Loss가 0.263까지 줄어들었다.  

**어라..?**  
단순한 우연인 걸까. 실험을 각각 5번 반복해봤다.  

$\sqrt{d_k}$를 사용했을 때는 0.31 아래로 단 한번도 내려가지 못한 반면  
$d_k$를 사용했을 때는 0.28 위로 올라간적이 한번도 없다.

또한 loss 수렴 속도도 $d_k$를 사용했을 때가 더 빠르다.

조금 더 신뢰도를 확보하기 위해 chatbot 데이터 뿐만 아니라 독일어-영어 번역 데이터에 대해서도 적용해봤다. 결과는 똑같았다.

현재 사용하는 데이터와 파라미터 상으로는 $d_k$로 나누는 것이 좋다고 생각해도 되겠다.

#### 이유가 뭘까

사실 실험이 너무 제한적이고 valid set을 사용하지도 않았으며 결과가 정확하다고 할 수 있는지 확신이 서진 않는다. 
그래도 끄적여보자면,  

softmax의 미분값은 해당 값이 0.5일때 최대값을 가진다. 그래서 gradient vanishing을 최대한 줄이려면 0 이나 1 가까이보다는 0.5 근처에 모여있는게 좋다.  
하지만 softmax는 모든 합이 1이므로 $d_k$의 값이 커질수록 각각의 값은 필연적으로 0에 가까워질 수 밖에 없다.  

그래서 각각을 softmax에 넣기 전 각각의 값들을 적당히 평탄하게 만들어주는게 좋다. 그렇다고 너무 평평해지면 미분값이 커야할 값들이 별다른 차별점을 가지지 못하게 되는 수도 있으니 scaling에는 적당한 값이 필요할 것이다. 그 적절한 값이 정확히 어디인지는 모르겠지만 현재 상황에서는 $\sqrt{d_k}$ 보다는 $d_k$에 가깝지 않았나 하는 생각이다.

번외로 $\sqrt{d_k}$ 도 $d_k$ 도 아닌 훨씬 더 큰값인 $d_k^2$ 으로 나누어 scaling 해주는 실험도 몇번 반복 해봤다.  
결론부터 말하면 현재 상황에서 $\sqrt{d_k}$ 보다는 성능이 좋았고 $d_k$ 보다는 근소하게 비슷하거나 좋지 않았다.  

수학적으로 아름답다고 느낄만한 최적의 scaling 값이 있을텐데...  
아니면 지금이 정답인데 뭔가 놓치고 있는 것일까.. 궁금하다...!  

**softmax 미분 참고**  
$\mathbf{Let} \; a = \dfrac{e^{z_1}}{e^{z_1}+e^{z_2}+e^{z_3}}\;, \quad \dfrac{\partial a}{\partial z_1} = a(1-a)$

Scaled Dot-Product Attention의 scaling에 대해서는 여기서 마무리하고 Input과 Target의 embedding 벡터에 $\sqrt{d_{model}}$를 곱하는 부분으로 넘어가보자.

## Embedding Vector Scaling

[Chatbot 만들기 (1)의 `Encoder`와 `Decoder`](https://fidabspd.github.io/2022/02/23/transformer_chatbot-1.html#encoder-stacks)의 embedding 부분을 보자.

```python
self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)
...
emb = self.tok_emb(x) * self.scale + self.pos_emb(pos)
```

token embedding의 결과에 $\sqrt{d_{model}}$을 곱하고 position embedding과 더한다.  
여기의 $\sqrt{d_{model}}$을 곱하는 이유가 뭘까.

마찬가지로 두가지로 나눠서 생각하자.

1. 왜 scaling을 하는지.
2. 그 값이 왜 하필 $\sqrt{d_{model}}$인지.

### scaling이 필요한 이유

논문부터 읽어보자.

>In the embedding layers, we multiply those weights by $\sqrt{d_{model}}$ .

**음...? 끝...?**  

놀랍게도 끝이다. 바로 앞 문장에 뭐라뭐라 써있긴 하지만 $\sqrt{d_{model}}$을 통한 scaling의 근거는 아니다.  

혹시 내가 해석을 잘못한걸까? 구글링도 해보자.

![embedding_scale_question_0.png](/assets/img/posts/analysis_for_transformer/embedding_scale_question_0.png)

![embedding_scale_question_1.png](/assets/img/posts/analysis_for_transformer/embedding_scale_question_1.png)

비슷한 의구심을 품은 질문들이 보인다. 해석을 잘못한건 아닌 모양이다.  
다행이다. ~~다행이 아닌가..? 논문에는 왜 이유를 써놓지 않았을까~~

그럼 대체 scaling을 왜 했단 말인가.

위 질문에 대해 이 [링크](https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec)와 함께 이런 답변이 달려있다.

>The reason we increase the embedding values before the addition is to make the positional encoding relatively smaller. This means the original meaning in the embedding vector won’t be lost when we add them together.

오호. **positional encoding이 더해지면서 token embedding의 의미가 약해지는걸 방지하기 위함이다.** (나는 positional embedding을 사용하긴 했지만)  
굉장히 설득력 있다.

진짜인지 궁금하다. 마찬가지로 실험해보자.  

우선 $\sqrt{d_{model}}$을 곱하는 scaling을 빼보자.

#### Embedding Scaling 제거

```
------------------------------
Epoch: 01
Train Loss: 41.257
Epoch Time: 0m 9s
------------------------------
Epoch: 02
Train Loss: 35.059
Epoch Time: 0m 9s
------------------------------
Epoch: 03
Train Loss: 33.479
Epoch Time: 0m 9s

...

------------------------------
Epoch: 48
Train Loss: 6.011
Epoch Time: 0m 9s
------------------------------
Epoch: 49
Train Loss: 5.898
Epoch Time: 0m 9s
------------------------------
Epoch: 50
Train Loss: 5.811
Epoch Time: 0m 9s
```

기본적으로 0.3 정도까지는 기본으로 내려가던 loss가 5.8까지 치솟았다.

**positional encoding이 더해지면서 token embedding의 의미가 약해지는걸 방지하기 위함**이 맞는듯 보인다.

그렇다면 또 한가지 궁금한점.

token embedding의 영향력이 줄어들지 않게 하기 위함이라면 token embedding은 그대로 두고 positional encoding(positional embedding)을 $\sqrt{d_{model}}$로 나누어 scaling하면 어떨까?  
이것도 실험해보자

#### Positional Embedding Scaling

코드는 단순하게 이렇게 바꿨다.

```python
emb = self.tok_emb(x) + self.pos_emb(pos) / self.scale
```

결과

```
------------------------------
Epoch: 01
Train Loss: 40.652
Epoch Time: 0m 9s
------------------------------
Epoch: 02
Train Loss: 34.497
Epoch Time: 0m 9s
------------------------------
Epoch: 03
Train Loss: 31.683
Epoch Time: 0m 9s

...

------------------------------
Epoch: 48
Train Loss: 0.423
Epoch Time: 0m 9s
------------------------------
Epoch: 49
Train Loss: 0.418
Epoch Time: 0m 9s
------------------------------
Epoch: 50
Train Loss: 0.425
Epoch Time: 0m 9s
```

기존 loss인 0.3정도와 비교하면 많은 차이가 나지만 scaling을 아예 없앴던 결과보다는 훨씬 낫다.

이로써 더욱 확실해졌다.  

**scaling을 하는 이유는 token embedding의 영향력이 줄어드는 것을 막기 위함이다.**

그럼 또다시. 

### 왜 하필 $\sqrt{d_{model}}$ 일까

이건 깊게 생각할 필요는 없겠다. 앞서 말한 내용의 반복이다.

**그냥 Token Embedding 벡터 길이의 기대값을 곱하는 컨셉인 것이다.**

사실 token embedding에 $\sqrt{d_{model}}$을 곱하는 것보다 앞서 실험했던 [Positional Embedding Scaling](#positional-embedding-scaling)에서 한 것 처럼 positional embedding을 $\sqrt{d_{model}}$로 나누는게 결과가 더 좋지 않을까? 하는 기대가 있었다.  

그 이유는 token embedding lookup table을 업데이트하는 가중치가 아니라 그냥 input값으로 본다면?  
input값의 scale이 전체적으로 크면 통상적으로 모델 loss 수렴에 도움이 되지 않는다. (weight initialize를 어떻게 하느냐에 따라 다르겠지만 첫 batch의 loss가 global minima에서 한참 먼 값이 되어버리는 경우가 많다. 아예 수렴하지 못하고 우주로 가버리는 경우도 있다.)  
그래서 굳이 멀쩡한 scale을 키우는 것보단 줄이는게 낫지 않을까 싶었는데 결과는 아니었다.

그래도 token embedding의 영향력이 줄어드는 것을 막기 위함이라는 컨셉에 큰 차이는 없으니 '[Positional Embedding Scaling](#positional-embedding-scaling)에도 활로가 있지 않을까?'하는 생각에 실험을 한가지 더 해봤다.

바로 Scaled Dot-Product Attention에서 실험했던 [$d_k$를 이용한 scaled attention](#d_k-로-나누기)을 결합하는 것이다.  

마찬가지로 실험해보자.

```
# of trainable parameters: 11,818,450
graph already exists
------------------------------
Epoch: 01
Train Loss: 40.595
Epoch Time: 0m 9s
------------------------------
Epoch: 02
Train Loss: 33.945
Epoch Time: 0m 9s
------------------------------
Epoch: 03
Train Loss: 29.974
Epoch Time: 0m 9s

...

------------------------------
Epoch: 48
Train Loss: 0.324
Epoch Time: 0m 9s
------------------------------
Epoch: 49
Train Loss: 0.291
Epoch Time: 0m 9s
------------------------------
Epoch: 50
Train Loss: 0.304
Epoch Time: 0m 9s
```

기존에 아무것도 건드리지 않은 Transfomer 그대로 사용한 결과인 0.32보다는 좋은 결과가 나왔다.  
token embedding의 영향력이 줄어드는 것을 막기 위해 positional embedding의 scale을 줄인다는 생각이 틀리지는 않았나보다.

그런데 positional embedding의 scaling과 scaled attention의 방식을 바꾼 것을 같이 실행했을 때 어떤 효과가 났기에 positional embedding 단독 사용보다 효과가 좋았을까의 측면으로 본다면 둘의 상호작용에는 확신이 들지 않는다.

그저 scaled positional embedding과 scaled attention의 방식을 바꾸는 것 둘을 함께하면 $QK^T$ 행렬의 scale이 전체적으로 더욱 작아질텐데 그게 효과가 있지 않을까... 하는 정도로 어림짐작간다. 마찬가지로 이 방법에 대해서도 최적의 scaling이 있을텐데 현재로써는 명확하게 보이진 않는지라 아쉽다.

## 마무리

Transformer의 scaling에 대해 깊이 고민하고 탐구해봤다.  

뭔가 시원하게 떨어지는 결론을 얻진 못해서 (특히 Token Embedding에 대해서는 더더욱) 조금 찝찝구리 하지만 혼자 생각해보면서 여러 깨달음을 얻기도 했고, 그런 깨달음이 실제 결과물로 나타날 수도 있음을 눈으로 확인했다.  
그런점에서 의미있게 느껴진다.

Transformer에 대해서는 충분히 깊게 고민했으니 이제 계속해서 TTS를 향해 달려보자.
