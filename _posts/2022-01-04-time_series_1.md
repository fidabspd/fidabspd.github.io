---
layout: post
title: Time Series - 시계열 데이터의 기본적인 특징과 간단한 모델
tags: [Time-Series, Tensorflow, Keras]
excerpt_separator: <!--more-->
---

시계열 데이터를 다루고 TensorFlow로 모델을 구성하는 데 있어 겪는 문제들과 그에 대한 가이드를 시리즈 형식으로 정리해보려 한다.  
기본적인 시계열 데이터의 특징을 알아보고 간단한 모델을 만드는 것으로 시계열 시리즈의 첫번째를 시작한다. <!--more-->
아주 간단한 시계열 모델에 점차 살을 붙여가는 형식으로 진행될 예정이며 목차는 다음과 같다.

## 목차

1. **시계열 데이터의 기본적인 특징과 간단한 모델**
1. tf.data.Dataset을 이용한 시계열 데이터 구성
1. Multi-Input 시계열 모델
1. Multi-Task Learning 시계열 모델 (1)
1. Multi-Task Learning 시계열 모델 (2)
1. 시계열 target의 결측
1. 이전의 예측값을 다음의 input으로 recursive하게 이용

## 원본 코드 ➞ [<span style="color:#AC1538">CODE (GitHub)</span>](https://github.com/fidabspd/time_series/tree/master/codes/1_super_basic.ipynb)

***

## 시작하기 전에

- **본 포스팅 시리즈의 목적은 시계열 예측의 정확성을 높이기 위함이 아니다.**
- **단지 시계열 데이터를 다루고 예측함에 있어 통상적으로 부딪히는 여러 문제들과 그를 어떻게 해결하면 좋을 지 가이드를 제시하는 것에 목적이 있다.**
- **데이터는 [DACON 전력사용량 예측 AI 경진대회](https://dacon.io/competitions/official/235736/overview/description)의 데이터를 사용한다.** 감사합니다 DACON!
    - 본 competition은 60개의 건물에 대해 전력 수요량을 미리 예측하는 대회이다.
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
    - **이번 포스팅에서는 `num`이 1인 건물의 `target` 만을 사용한다.**

***

## 시계열 데이터의 특징

### 대부분 정형 데이터의 특징

머신러닝 공부를 시작하는 단계에서 접하는 대부분의 정형 데이터들은 하나의 row가 하나의 데이터를 나타내는 경우가 많다. (편의상 앞으로 이런 데이터를 OneRowOneData를 축약하여 **OROD**이라고 부르도록 하겠다. ~~필자가 방금 멋대로 정한 단어다.~~) 튜토리얼로 많이 사용되는 캐글의 타이타닉 생존자 예측 데이터를 보면 하나의 row가 한명의 승객정보를 담고 있고, 이를 통해 해당 승객이 생존했는지 생존하지 못했는지 예측한다. 하나의 row에 예측을 위해 사용될 input과 예측 target 값이 모여있다.  

### 시계열 데이터의 One Row

시계열 데이터는 하나의 row가 하나의 데이터를 나타내지 않는다. 시간의 흐름에 따라 row 방향으로 데이터를 쌓는다면 이전 row들을 이용해 다음 row들을 예측한다.  
본인이 하나의 row가 하나의 데이터를 나타내는 데이터만 다룰 수 있는 상황이라 한들, 시계열 데이터를 다룰 수 없는 것은 아니다. 다루기 익숙한 모양(하나의 row가 하나의 데이터)에 맞춰 데이터를 재구성 하면 된다.

### 시계열 데이터 OROD로 만들기

앞서 말한 OROD데이터는 input과 target이 정해져있다. (물론 파생변수를 만들거나 target에 변형을 가하면 달라지긴 한다.) 하지만 시계열 데이터는 input과 target에 대한 자유도가 상당히 높다. 이를 확실하게 구성하기 위해서는 다음과 같은 내용을 미리 정해야한다.

- 이전 몇 time 을 사용하여 다음 값을 예측할 것인지
- 다음 몇 time 동안의 값을 예측할 것인지
- 예측을 몇 time 단위로 띄어가며 진행할 것인지 (데이터의 양 조절)

그리고 각각을 앞으로 다음과 같이 칭하기로 한다. (완전히 약속된 단어는 아니지만 통상 이렇게 지칭하면 의사소통에 문제는 없다.)

- window_size
- target_length
- shift

이들은 일종의 hyper parameter로 볼 수 있으며 상황과 데이터에 맞게 조절해야한다.

#### window_size, target_length, shift 예시

비트코인의 가격이 분단위로 기록된 10일치 데이터를 가지고 있다고 가정해보자.   
이때 ```window_size = 120, target_length = 10, shift = 5```를 다른말로 하면  
```이전 2시간의 데이터를 이용하여 이후 10분을 예측하며 5분 단위로 데이터를 구분한다.```
라고 할 수 있다.

#### 총 row 수

위의 예시를 이용하여 다음 시계열 데이터를 OROD데이터로 구성한다면 몇개의 row가 생기게 될까.  
이는 다음과 같이 계산 할 수 있다.

```
총 seqence길이(10일치): 10*24*60 = 14400
window_size + target_length: 120 + 10 = 130
총 row 수: {(14400 - 130) / 5} + 1 = 2855
```

일반화 해보자면

```
총 row 수: ((seqence_length - (window_size+target_lenght)) / shift) + 1
```

이렇게 표현이 가능하다.  
혹시라도 ```((seqence_length - (window_size+target_lenght))```가 ```shift```로 나눠 떨어지지 않는 것이 걱정이라면 각 숫자를 조정하거나, 앞부분 데이터를 조금 잘라내거나, 딱 한곳의 shift만 임의로 조정함으로서 해결할 수 있다.

## CODE

### Libraries

```python
from copy import deepcopy

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
```

앞으로 사용하는 모든 데이터는 [DACON 전력사용량 예측 AI 경진대회](https://dacon.io/competitions/official/235736/overview/description)의 전력 사용량 데이터를 가공하여 사용한다.  
데이터는 다음과 같다.  

```python
data_path = '../data/'

train_origin = pd.read_csv(data_path+'train.csv', encoding='cp949')
data = deepcopy(train_origin)
data.columns = [
    'num', 'date_time', 'target', 'temp', 'wind',
    'humid', 'rain', 'sun', 'non_elec_eq', 'sunlight_eq'
]
data = data.loc[data['num'] == 1, ['date_time', 'target']]
data
```

![data](/assets/img/posts/time_series_1/data.png)  
20년 6월 1일부터 20년 8월 24일까지 시간(hour)단위로 구성된 데이터

### train, valid, test 구성

간단하게 testset은 마지막 일주일인 8월 18일 ~ 8월 24일로 하고 이를 예측하는 모델을 구성해보기로 한다.  
그런데 validset은 어떻게 구성하면 좋을까.  
구성에 고려해야할 것은 여러가지가 있지만 크게 두가지정도만 짚도록 하자.

1. validset이 시계열 데이터의 주기를 모두 포함하였는지
2. 전체 데이터 중 어떤 데이터를 valid로 사용할 지

우선 **1번**고려사항에 관해서는, validset은 당연히 시계열 데이터의 주기를 모두 포함하는 것이 좋다.  
어떤 도로의 교통량을 예측하는 데 validset을 주말 시간대로만 구성한다면 주말만 잘 맞추는 모델이 될 것이 뻔하다.  
하지만 현실적으로 모든 주기를 모두 포함하기는 쉽지 않기 때문에 상황에 따른 적절한 분배가 필요하다.

**2번**고려사항에 관해서는, 데이터를 구성하는 방법은 크게 두가지 정도로 생각할 수 있으며 장단점은 다음과 같다.

1. train의 일부를 임의로 추출하여 valid로 사용한다.
    - 장점
        - test를 예측하기 위해 가장 중요한 데이터인 바로 전 기간을 학습에 사용할 수 있다.
    - 단점
        - valid score를 신뢰하기 힘들다.
            - 시간에 따라 경사가 급한 파도형태의 데이터가 아닌 이상 앞 뒤 데이터를 학습에 활용했다면 그 사이 데이터는 잘 맞출 수 밖에 없다.
2. test 바로 직전의 기간을 valid로 사용한다.
    - 장점
        - validset을 testset에 가장 근접하게 구성한 만큼 valid score를 신뢰할 수 있다.
    - 단점
        - testset에 가장 근접한 데이터를 학습에 활용할 수 없다.

    2번 방법을 사용하여 valid score에 대한 신뢰도를 높이면서 이를 train에도 활용하는 방법도 있긴 하다.  
    trainset만으로 모든 hyperparameter에 대한 tuning을 마치고 validset을 trainset에 포함시켜 다시 train하는 방법.  
    다만 이는 trainset만으로 hyperparameter를 tuning하였기 때문에 validset이 train에 포함되는 순간 결과가 이상해지는 경우가 생길 수 있다.  
    이런 이유로 필자는 이 방법을 그렇게 추천하지 않으며, 딥러닝 모델에는 특히 더욱 그렇다.

전체적인 흐름보다 주기에 훨씬 의존적인 모습을 보인다면 **1번**을 사용할 수도 있다. 다만 대부분의 경우에 **2번**방법을 추천한다.

**2번**방법을 사용하여 20년 8월 11일 ~ 8월 17일은 valid_data, 20년 8월 18 ~ 8월 24일일은 test_data, 나머지는 train_data로 사용하여 data를 OROD형태로 재구성 해보자.  
(monthly 주기는 일단은 고려하지 않는다.)  

### Scaling

우선 train에 사용될 데이터만을 이용해 기본적인 standard scaling을 해준다.  
(추후 모델 metric 중 하나로 inverse된 target을 이용하여 rmse를 계산할 것이기 때문에 평균과 표준편차는 따로 저장)

```python
def mk_mean_std_dict(data):
    mean_std_dict = {
        col: {
            'mean': data[col].mean(),
            'std': data[col].std()
        } for col in data.columns
    }
    return mean_std_dict


scaling_cols = ['target']
mean_std_dict = mk_mean_std_dict(data[scaling_cols][:CONFIGS['valid_start_index']])
CONFIGS['mean_std_dict'] = mean_std_dict


def standard_scaling(data, mean_std_dict=None):
    if not mean_std_dict:
        mean_std_dict = mk_mean_std_dict(data)
    new_data = data.copy()
    for col in new_data.columns:
        new_data[col] -= mean_std_dict[col]['mean']
        new_data[col] /= mean_std_dict[col]['std']
    return new_data


data[scaling_cols] = standard_scaling(data[scaling_cols], mean_std_dict)
```

### OROD

다음과 같은 `window_size`, `target_length`, `shift`를 이용하여 데이터를 구성한다.

- ```window_size```: 7*24 (7일)
- ```target_length```: 3 (3시간)
- ```shift```: 1 (1시간)

```python
data = data[['target']]

CONFIGS = {
    'valid_start_index': 1704,
    'test_start_index': 1872,
    
    'window_size': 7*24,
    'target_length': 3,
}
input_cols = [f't-{i}' for i in range(CONFIGS['window_size'], 0, -1)]
target_cols = [f't+{i}' for i in range(CONFIGS['target_length'])]
CONFIGS['input_cols'] = input_cols
CONFIGS['target_cols'] = target_cols


def mk_time_series(data):
    
    new_data_length = data.shape[0]-CONFIGS['window_size']-CONFIGS['target_length']+1
    new_data_shape = (new_data_length, CONFIGS['window_size']+CONFIGS['target_length'])
    new_data = np.zeros(new_data_shape)

    for i in range(new_data_length):
        new_data[i, :CONFIGS['window_size']] = data['target'][i:i+CONFIGS['window_size']]
        new_data[i, CONFIGS['window_size']:] = \
            data['target'][i+CONFIGS['window_size']:i+CONFIGS['window_size']+CONFIGS['target_length']]

    new_data = pd.DataFrame(new_data)
    new_data.columns = input_cols + target_cols
    
    return new_data


def split_data(data, CONFIGS):
    
    train = data[:CONFIGS['valid_start_index']]
    valid = data[
        CONFIGS['valid_start_index']-CONFIGS['window_size']:\
        CONFIGS['valid_start_index']+CONFIGS['test_lenght']
    ]
    test = data[
        CONFIGS['test_start_index']-CONFIGS['window_size']:\
        CONFIGS['test_start_index']+CONFIGS['test_lenght']
    ]
    
    train, valid, test = \
        mk_time_series(train), mk_time_series(valid), mk_time_series(test)
    
    return train, valid, test

train, valid, test = split_data(data, CONFIGS)
```

재구성 된 데이터는 다음과 같다.

```python
train
```

![train](/assets/img/posts/time_series_1/train.png)  

```python
seqence_length = 2040 - 24*7*2  # valid와 test를 빼줬기 때문에 -24*7*2
window_size = 24*7
target_lenght = 3
shift = 1

nrow = ((seqence_length - (window_size+target_lenght)) / shift) + 1
print(f'총 row 수: {nrow}')
# 총 row 수: 1534
```

계산대로 총 1534개의 row가 생긴 모습이 보인다.

### Modeling

모델은 아주 간단한 NN 모델을 만들고 훈련해보자.  

우선 model metric으로 사용할 inverse된 target을 이용한 rmse를 계산하는 함수를 만든다.

```python
def inversed_rmse(y_true, y_pred, mean, std):
    y_true = y_true*std+mean
    y_pred = y_pred*std+mean
    mse = tf.reduce_mean((y_true-y_pred)**2)
    return tf.sqrt(mse)

inversed_rmse_metric = lambda y_true, y_pred: inversed_rmse(y_true, y_pred, **CONFIGS['mean_std_dict']['target'])
```

참고로 model의 `compile`에 loss나 metric에 사용되는 함수의 경우 그 안의 계산 과정에서 텐서형을 계속 유지해야한다.  
따라서 numpy연산등을 사용하면 에러가 발생한다.

#### Set Model

```python
def set_model(CONFIGS, model_name = None, print_summary=False):
    inputs = Input(batch_shape=(None, CONFIGS['window_size']), name='inputs')
    dense_0 = Dense(64, activation='relu', name='dense_0')(inputs)
    dense_1 = Dense(32, activation='relu', name='dense_1')(dense_0)
    outputs = Dense(CONFIGS['target_length'], name='outputs')(dense_1)
    
    if not model_name:
        model_name = CONFIGS['model_name']
    
    model = Model(
        inputs, outputs,
        name = model_name
    )
    
    optimizer = Adam(learning_rate=CONFIGS['learning_rate'])
    model.compile(
        loss = MeanSquaredError(),
        optimizer = optimizer,
        metrics=[inversed_rmse_metric],
    )
    
    if print_summary:
        model.summary()
    
    return model


CONFIGS['model_name'] = 'super_basic'
CONFIGS['learning_rate'] = 1e-4
model = set_model(CONFIGS, print_summary=True)
```

### Train

```python
def train_model(model, train, valid, CONFIGS):
    
    X_train, y_train = train[CONFIGS['input_cols']], train[CONFIGS['target_cols']]
    X_valid, y_valid = valid[CONFIGS['input_cols']], valid[CONFIGS['target_cols']]
    
    early_stop = EarlyStopping(
        patience=CONFIGS['es_patience']
    )
    save_best_only = ModelCheckpoint(
        filepath = f'{CONFIGS["model_path"]}{CONFIGS["model_name"]}.h5',
        monitor = 'val_loss',
        save_best_only = True,
        save_weights_only = True
    )
    
    history = model.fit(
        X_train, y_train,
        batch_size = CONFIGS['batch_size'],
        epochs = CONFIGS['epochs'],
        validation_data = (X_valid, y_valid),
        callbacks = [
            early_stop,
            save_best_only,
        ]
    )
    
    return history


CONFIGS['model_path'] = '../model/'
CONFIGS['batch_size'] = 64
CONFIGS['epochs'] = 100
CONFIGS['es_patience'] = 10

history = train_model(model, train, valid, CONFIGS)
```

![model_summary](/assets/img/posts/time_series_1/model_summary.png)  

### Evaluate

best model을 load하여 test를 예측하고 성능을 비교해보자.

```python
best_model = set_model(CONFIGS, model_name='best_'+CONFIGS['model_name'])
best_model.load_weights(f'{CONFIGS["model_path"]}{CONFIGS["model_name"]}.h5')

X_train, y_train = train[CONFIGS['input_cols']], train[CONFIGS['target_cols']]
X_valid, y_valid = valid[CONFIGS['input_cols']], valid[CONFIGS['target_cols']]
X_test, y_test = test[CONFIGS['input_cols']], test[CONFIGS['target_cols']]

y_train_pred = best_model.predict(X_train)
y_valid_pred = best_model.predict(X_valid)
y_test_pred = best_model.predict(X_test)

train_loss, train_rmse = best_model.evaluate(X_train, y_train, verbose=0)
valid_loss, valid_rmse = best_model.evaluate(X_valid, y_valid, verbose=0)
test_loss, test_rmse = best_model.evaluate(X_test, y_test, verbose=0)

print(f'train_loss: {train_loss:.6f}\ttrain_rmse: {train_rmse:.6f}')
print(f'valid_loss: {valid_loss:.6f}\tvalid_rmse: {valid_rmse:.6f}')
print(f'test_loss: {test_loss:.6f}\ttest_rmse: {test_rmse:.6f}')

# train_loss: 0.048540	train_rmse: 25.183891
# valid_loss: 0.142397	valid_rmse: 45.505894
# test_loss: 0.125447	test_rmse: 46.547405
```

데이터 구성부터 간단한 모델링까지 끝 마쳤다.
**(모델 성능을 높이기 위한 고민을 담은 포스팅은 아니기 때문에 당장 모델 성능은 신경쓰지 않는다.)**

처음부터 정독했다면 이런저런 의문점들이 생기는 부분이 있을 것이다. 예상되는 것들을 적어보자면

- 매번 데이터 구성을 저렇게 매뉴얼하게 할 것인가?
- testset의 시간은 24시간인데 target_length를 3으로 구성하면 알지 못한다고 가정한 testset의 target이 input으로 들어간거 아닌가?

등등 외의 여러개가 있을 것으로 예상된다.  

모든 의문점을 해결할 수는 없겠지만 이런 의문들에 대한 내용은 목차에 따라 이어질 포스팅에서 다룰 예정이다.

## 목차

1. **시계열 데이터의 기본적인 특징과 간단한 모델**
1. tf.data.Dataset을 이용한 시계열 데이터 구성
1. Multi-Input 시계열 모델
1. Multi-Task Learning 시계열 모델 (1)
1. Multi-Task Learning 시계열 모델 (2)
1. 시계열 target의 결측
1. 이전의 예측값을 다음의 input으로 recursive하게 이용

## 원본 코드 ➞ [<span style="color:#AC1538">CODE (GitHub)</span>](https://github.com/fidabspd/time_series/tree/master/codes/1_super_basic.ipynb)
