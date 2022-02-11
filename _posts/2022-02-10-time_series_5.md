---
layout: post
title: Time Series - 시계열 target의 결측
tags: [Time-Series, Tensorflow, Keras]
excerpt_separator: <!--more-->
---


<!--more-->

## 목차

1. 시계열 데이터의 기본적인 특징과 간단한 모델
1. tf.data.Dataset을 이용한 시계열 데이터 구성
1. Multi-Input 시계열 모델
1. Multi-Task Learning 시계열 모델 (1)
1. Multi-Task Learning 시계열 모델 (2)
1. **시계열 target의 결측**
1. 이전의 예측값을 다음의 input으로 recursive하게 이용

## 원본 코드 ➞ [<span style="color:#AC1538">CODE (GitHub)</span>](https://github.com/fidabspd/time_series/blob/master/codes/5_handling_missing_value.ipynb)

***

## 시작하기 전에

- **본 포스팅 시리즈의 목적은 시계열 예측의 정확성을 높이기 위함이 아니다.**
- **단지 시계열 데이터를 다루고 예측함에 있어 통상적으로 부딪히는 여러 문제들과 그를 어떻게 해결하면 좋을 지 가이드를 제시하는 것에 목적이 있다.**
- **데이터는 [DACON 전력사용량 예측 AI 경진대회](https://dacon.io/competitions/official/235736/overview/description)의 데이터를 사용한다.** 감사합니다 DACON!
    - 60개의 건물의 전력 수요량을 미리 예측하는 것이 목적이며, 본 대회는 그를 위한 데이터를 다음과 같이 제공한다.
    - 데이터 column 구성
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
    - **이번 포스팅에서는 모든 데이터를 사용**

***
