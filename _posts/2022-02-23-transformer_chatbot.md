---
layout: post
title: Transformer - 챗봇 만들기
tags: [Chatbot, NLP, Transformer, PyTorch]
excerpt_separator: <!--more-->
use_math: true
---

과거 TTS 프로젝트를 진행해본 적이 있다. 해당 프로젝트 당시 **carpedm20** 님의 [multi-speaker-tacotron-tensorflow](https://github.com/carpedm20/multi-speaker-tacotron-tensorflow)를 참고하여 사용했고, 꽤나 의미있는 결과물을 만들어냈다. 나름의 소스 리뷰를 거쳤고 데이터 수집부터 직접 녹음한 목소리를 함께 train하여 결과물을 만들기도 했지만 아쉬움이 남는다.<!--more--> 그 이유는 '이 프로젝트를 100% 이해하고 있나?'라는 질문에 자신있게 'Yes!'라고 대답하지 못하겠다. 물론 해당 프로젝트에 대한 질문을 받으면 대답하는데 무리는 없다. 하지만 이정도로 만족하지 못하겠다. 그런 의미에서 기존의 아키텍쳐를 답습하여 그대로 만들기보단, 기존의 아키텍쳐를 발전시켜 새로운 TTS를 만들어보자.

## How To Improve

기존에 사용했던 Deep Voice 2 아키텍쳐이다. ([Deep Voice 2: Multi-Speaker Neural Text-to-Speech](https://arxiv.org/abs/1705.08947))

![tacotron](/assets/img/posts/transformer_chatbot/tacotron.png)

개선하고자 하는 Key는 `seq2seq -> transformer`이다.

기존에 사용했던 아키텍쳐는 seq2seq에 기반을 두고 있다. 정확히는 인코더와 디코더에 어텐션을 얹어 사용하고있다. 그리고 이것을 Transformer 형태로 바꾸고 싶다.
사실 Tacotron 이후 Glow-TTS, FastSpeech 등에서 Transformer 기반의 아키텍쳐를 사용하고 있음은 알고있다. (자세히 알고있는 내용은 아니라 말하기가 좀 조심스럽다.)  
그래서 이런 최신 아키텍쳐를 공부하고 구현하는 방식을 택해볼까 싶기도 했지만, 이번에는 나만의 연구를 해보고 싶었다. 최신 SOTA를 놔두고 Tacotron 개선 연구는 어쩌면 단순한 뒷북치기일지도 모르지만, SOTA를 참고하지 않고 나만의 연구를 한다는 점에서 분명 배우는 것이 있을 것이라 생각한다.

## Transformer

순서대로라면 데이터 수집부터 시작해야겠지만 *구조를 이렇게 바꿀거다!*를 선언해버린 이상 Transformer를 지금 짚지 않고 넘어가는 것도 좀 웃기다.  
어차피 Transformer를 만들어 씌우는게 목적이니, 일단은 Transformer를 이용한 챗봇을 만들어보자. 그 뒤에 데이터 수집부터 시작한다. (솔직히 재미있을 것 같아서 빨리 하고싶었다..!)

우선은 Transformer를 구현해보자.

### Architecture

![transformer_architecture](/assets/img/posts/transformer_chatbot/transformer_architecture.png)

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

![multi_head_attention](/assets/img/posts/transformer_chatbot/multi_head_attention.png)

- `Multi-Head Attention`은 `Scaled Dot-Product Attention`이 `h`번 반복된 구조이다.
- `Scaled Dot-Product Attention`은 `Linear`를 지난 `Q`, `K`, `V`를 이용한다.  
(`Scaled Dot-Product Attention`에 대한 설명은 생략)
- `Scaled Dot-Product Attention`의 결과들을 `Concat`하여 `Linear`를 이용하여 최종 output을 만든다.

`Scaled Dot-Product Attention`의 계산은 한줄의 식으로 표현이 가능하다.  

$Attention(Q, K, V) = softmax\bigg(\frac{QK^T}{\sqrt{d_k}}\bigg)V$

