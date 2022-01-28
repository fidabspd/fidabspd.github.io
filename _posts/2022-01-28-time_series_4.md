---
layout: post
title: Time Series - Multi-input 시계열 모델
tags: [Time-Series, Tensorflow, Keras]
excerpt_separator: <!--more-->
---

앞선 모든 시계열 시리즈에서는 `num == 1`인 데이터만 사용하여 데이터와 모델을 구성하였습니다. 즉 여러개의 sequence를 사용하지 않았습니다.  
하지만 주어진 데이터는 60개의 건물 모두에 대한 sequence가 데이터로 주어졌습니다. 이 모두를 이용하여 데이터와 모델을 구성해보겠습니다.
<!--more-->

## 목차

1. 시계열 데이터의 기본적인 특징과 간단한 모델
1. tf.data.Dataset을 이용한 시계열 데이터 구성
1. Multi-input 시계열 모델
1. **Multi-Task Learning 시계열 모델**
1. 시계열 target의 결측
1. 이전의 예측값을 다음의 input으로 recursive하게 이용

## 원본 코드 ➞ [<span style="color:#AC1538">CODE (GitHub)</span>](https://github.com/fidabspd/time_series/tree/master/codes/4_multi_task_learning.ipynb)

***

## 시작하기 전에

- **본 포스팅 시리즈의 목적은 시계열 예측의 정확성을 높이기 위함이 아닙니다.**
- **단지 시계열 데이터를 다루고 예측함에 있어 부딪히는 여러 문제들과 그를 어떻게 해결하면 좋을 지 가이드를 제시하기 위함입니다.**
- **데이터는 [DACON 전력사용량 예측 AI 경진대회](https://dacon.io/competitions/official/235736/overview/description)의 데이터를 사용합니다.** 감사합니다 DACON!
    - 본 데이터가 주어진 competition은 60개의 건물에 대해 전력 수요량을 미리 예측하는 대회입니다.
    - 데이터 column 구성은 다음과 같습니다.
        - `num`: 건물 번호
        - `date_time`: 데이터가 측정된 시간 (1시간 단위)
        - `target`: 전력 사용량 (예측해야하는 값)
        - `temp`: 기온(°C)
        - `wind`: 풍속(m/s)
        - `humid`: 습도(%)
        - `rain`: 강수량(mm)
        - `sun`: 일조(hr)
        - `non_elec_eq`: 건물별 비전기 냉방 설비 운영 여부 (1: 보유, 0: 미보유)
        - `sunlight_eq`: 건물별 태양광 설비 보유 여부 (1: 보유, 0: 미보유)
    - **이번 포스팅에서는 모든 데이터를 사용합니다.**

***

## Multi Task Learning

**Multi Task Learning**: Multi-task learning (MTL) is a subfield of machine learning in which multiple learning tasks are solved at the same time, while exploiting commonalities and differences across tasks. This can result in improved learning efficiency and prediction accuracy for the task-specific models, when compared to training the models separately. [출처: wikipedia](https://en.wikipedia.org/wiki/Multi-task_learning)

위키피디아의 설명과 같이 Multi Task Learning이란 관계가 있는 여러 학습을 동시에 수행하며, 개별적으로 훈련할때 보다 높은 효율성과 정확성을 기대할 수 있습니다.
