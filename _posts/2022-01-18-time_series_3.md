---
layout: post
title: Time Series - Multi-input 시계열 모델
tags: [Time-Series, Tensorflow, Keras]
excerpt_separator: <!--more-->
---

앞선 두개의 시계열 포스팅 시리즈에서는 target을 예측함에 있어 이전 시간대의 target만을 이용해 데이터를 구성하고 예측을 진행하였습니다.  
하지만 가지고 있는 데이터는 target 외에 기온, 풍속, 습도, 강수량, 날짜정보 등을 함께 제공합니다. <!--more-->  
이들을 함께 input으로 사용하여 모델을 구성해보겠습니다.  

## 목차

1. 시계열 데이터의 기본적인 특징과 간단한 모델
1. tf.data.Dataset을 이용한 시계열 데이터 구성
1. **Multi-input 시계열 모델**
1. Multi-Task Learning 시계열 모델
1. 시계열 target의 결측
1. 이전의 예측값을 다음의 input으로 recursive하게 이용

## 원본 코드 ➞ [<span style="color:#AC1538">CODE (GitHub)</span>](https://github.com/fidabspd/time_series/tree/master/codes)

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
    - **이번 포스팅에서는 `num`이 1인 건물의 데이터만을 사용합니다.**

***

## 시계열 데이터의 multi-input

[시계열 포스팅 시리즈의 첫번째 글](https://fidabspd.github.io/2022/01/04/time_series_1.html)에서 다뤘던 하나의 row가 하나의 data를 나타내는 데이터의 경우 multi-input이라고 해서 특별히 모델을 다른 형태로 만들어야 모델이 작동하진 않습니다. 하지만 시계열 데이터의 경우 multi-input으로 데이터를 구성하면 데이터의 형태가 일반적인 2d-array로는 구성할 수 없는 형태가 되기 때문에 flatten을 통해 데이터를 펴주거나, cnn이나 lstm 등의 모델을 이용해야합니다.

어째서 그런지는 [시계열 포스팅 시리즈의 첫번째 글](https://fidabspd.github.io/2022/01/04/time_series_1.html)에서 manual하게 구성했던 데이터를 보면 이해하기 쉽습니다.  

![train_manual](/assets/img/posts/time_series_3/train_manual.png)

위 데이터를 보면 target만으로 데이터를 구성해도 이미 하나의 row가 하나의 data를 나타내는 데이터의 multi-input과 데이터의 형태가 같습니다. 이에 target 외에 다른 시계열 데이터들을 함께 input으로 사용한다면 하나의 데이터마다 2d-array가 input으로 들어가는 데이터를 구성해야합니다.

input으로 활용할 수 있는 정보는 앞선 시간대의 시계열 정보 뿐만이 아닙니다. target date의 정보 또한 시계열 데이터 예측에 중요한 역할을 합니다.  
이 외에도 다양한 input들을 활용할 수 있겠지만 우선은 두가지 정보를 활용해보겠습니다.

![data](/assets/img/posts/time_series_3/data.png)

## Dataset

[시계열 포스팅 시리즈의 두번째 글](https://fidabspd.github.io/2022/01/10/time_series_2.html)에서 다뤘던 `tf.data`를 활용합니다.  
