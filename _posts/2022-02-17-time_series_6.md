---
layout: post
title: Time Series - Recursive Input Prediction
tags: [Time-Series, Tensorflow, Keras]
excerpt_separator: <!--more-->
---

앞선 시계열 시리즈들에서 validset과 testset의 sequence길이는 모두 일주일이었다. 하지만 `target_length`는 3에 불과했다. 계속 하던 방식처럼 trainset의 일부를 testset으로 분리하는 방식이 아닌 정말로 미래의 데이터를 예측해야하는 경우였다면 앞선 내용들은 세시간 뒤의 미래밖에 예측할 수 없다.  
<!--more-->
이를 일주일을 통째로 예측하기 위해서는 두가지 방법이 있다.

1. `target_length`를 일주일로 늘린다.
2. 모델의 예측값을 다시 input으로 넣어가며 recursive하게 데이터를 구성한다.

이중에서 1번 방법은 `target_length = 7*24` 해주면 그만이므로 다루지 않기로 하고, 2번 방법을 다뤄보도록 하자.

## 목차

1. 시계열 데이터의 기본적인 특징과 간단한 모델
1. tf.data.Dataset을 이용한 시계열 데이터 구성
1. Multi-Input 시계열 모델
1. Multi-Task Learning 시계열 모델 (1)
1. Multi-Task Learning 시계열 모델 (2)
1. 시계열 target의 결측
1. **이전의 예측값을 다음의 input으로 recursive하게 이용**

## 원본 코드 ➞ [<span style="color:#AC1538">CODE (GitHub)</span>](https://github.com/fidabspd/time_series/blob/master/codes/6_recursive_prediction.ipynb)

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

## Recursive Input Prediction

개념은 어려울 것이 전혀 없다. `window_size = 7; target_length = 1`을 이용하고 있다면,  
우선 t-7 ~ t-1를 input으로 이용하여 t0을 예측하고, 예측된 값인 t0를 다시 input에 포함시켜 t-6 ~ t0을 구성하고 이를 이용하여 t+1을 예측하면 된다.

참고로 대부분의 경우에는 이렇게 recursive하게 input을 활용하는 것(2번 방법) 보단 그냥 `target_length`자체를 늘려버리는 방법(1번 방법)이 성능면에서 더 낫다는 평가가 많으며 필자의 경험상도 그렇다.  
하지만 본 포스팅의 목적은 시계열 데이터를 이용하여 모델을 만드는 과정에서 마주하는 다양한 어려움들을 어떻게 해결하면 좋을지에 대해 정리하는 내용이기 때문에 **1번** 방법보다는 더 어려운 방법인 **2번** 방법을 써보도록 하자.

## CODE

### Libraries

```python
from copy import deepcopy

import numpy as np
import pandas as pd

import datetime

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.callbacks import TensorBoard, Callback
```

### Set Configs

```python
CONFIGS = {
    'data_path': '../data/',
    'model_path': '../model/',
    'model_name': 'recursive_prediction',
    'model_type': 'cnn1d',
    
    'dtype': tf.float32,
    
    'valid_start_date_time': '2020-08-11 00',
    'test_start_date_time': '2020-08-18 00',
    
    'buffer_size': 512,
    'batch_size': 64,
    'learning_rate': 1e-4,
    'epochs': 100,
    'es_patience': 10,
    
    'window_size': 7*24,
    'shift': 1,
    'target_length': 1,
}

CONFIGS['tensorboard_log_path'] = f'../logs/tensorboard/{CONFIGS["model_name"]}'
```

### Load Data

```python
train_origin = pd.read_csv(CONFIGS['data_path']+'train.csv', encoding='cp949')

data = deepcopy(train_origin)

data.columns = [
    'num', 'date_time', 'target', 'temp', 'wind',
    'humid', 'rain', 'sun', 'non_elec_eq', 'sunlight_eq'
]

data['num'] -= 1

CONFIGS['last_date_time'] = data['date_time'].max()
CONFIGS['n_buildings'] = len(data['num'].unique())
```

### 결측값 추가

### 시간 데이터 가공

### Make Building Info

### Scaling

### Visualize

### Fill Missing Value

### Dataset

#### Input Columns

#### Make tf.data.Dataset

### Modeling

### Customize Loss & Metric

#### Customize Layer

#### Set Model

#### Train

#### Evaluate

## 목차

1. 시계열 데이터의 기본적인 특징과 간단한 모델
1. tf.data.Dataset을 이용한 시계열 데이터 구성
1. Multi-Input 시계열 모델
1. Multi-Task Learning 시계열 모델 (1)
1. Multi-Task Learning 시계열 모델 (2)
1. 시계열 target의 결측
1. **이전의 예측값을 다음의 input으로 recursive하게 이용**

## 원본 코드 ➞ [<span style="color:#AC1538">CODE (GitHub)</span>](https://github.com/fidabspd/time_series/blob/master/codes/6_recursive_prediction.ipynb)
