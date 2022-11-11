---
title: Time Series - Multi-input 시계열 모델
tags: [Time-Series, Tensorflow, Keras]
---

앞선 두개의 시계열 포스팅 시리즈에서는 target을 예측함에 있어 이전 시간대의 target만을 이용해 데이터를 구성하고 예측을 진행했다.  
하지만 원본 데이터는 target 외에 기온, 풍속, 습도, 강수량, 날짜정보 등을 다른 다양한 데이터들 또한 함께 제공한다. 
이들을 함께 input으로 사용하여 모델을 구성해보자.

## 목차

1. [시계열 데이터의 기본적인 특징과 간단한 모델](https://fidabspd.github.io/2022/01/04/time_series_1.html)
1. [tf.data.Dataset을 이용한 시계열 데이터 구성](https://fidabspd.github.io/2022/01/10/time_series_2.html)
1. [**Multi-Input 시계열 모델**](https://fidabspd.github.io/2022/01/18/time_series_3.html)
1. [Multi-Task Learning 시계열 모델 (1)](https://fidabspd.github.io/2022/01/28/time_series_4-1.html)
1. [Multi-Task Learning 시계열 모델 (2)](https://fidabspd.github.io/2022/01/30/time_series_4-2.html)
1. [시계열 target의 결측](https://fidabspd.github.io/2022/02/10/time_series_5.html)
1. [Recursive Input Prediction](https://fidabspd.github.io/2022/02/17/time_series_6.html)

## 원본 코드 ➞ [<span style="color:#AC1538">CODE (GitHub)</span>](https://github.com/fidabspd/time_series/tree/master/codes/3_multi_input.ipynb)

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
    - **이번 포스팅에서는 `num`이 1인 건물의 데이터만을 사용한다.**

***

## 시계열 데이터의 multi-input

[시계열 포스팅 시리즈의 첫번째 글](https://fidabspd.github.io/2022/01/04/time_series_1.html)에서 다뤘던 하나의 row가 하나의 data를 나타내는 데이터의 경우 multi-input이라고 해서 특별히 모델을 다른 형태로 만들어야만 하는건 아니다. 하지만 시계열 데이터의 경우 multi-input으로 데이터를 구성하면 데이터의 형태가 일반적인 2D-array로는 구성할 수 없는 형태가 되기 때문에 flatten을 통해 데이터를 펴주거나, cnn이나 lstm 등의 모델을 이용해야한다.

어째서 그런지는 [시계열 포스팅 시리즈의 첫번째 글](https://fidabspd.github.io/2022/01/04/time_series_1.html)에서 manual하게 구성했던 데이터를 보면 이해하기 쉽다.  

![train_manual](/post_images/time_series_3/train_manual.png)

위 데이터를 보면 target만으로 데이터를 구성해도 이미 하나의 row가 하나의 data를 나타내는 데이터의 multi-input과 데이터의 형태가 같다. 이에 target 외에 다른 시계열 데이터들을 함께 input으로 사용한다면 하나의 데이터마다 2D-array가 input으로 들어가는 데이터를 구성해야한다.

input으로 활용할 수 있는 정보는 앞선 시간대의 시계열 정보 뿐만이 아니다. target time의 정보 또한 시계열 데이터 예측에 중요한 역할을 한다.  
이 외에도 다양한 input들을 활용할 수 있겠지만 우선은 두가지 정보를 활용해보자.

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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
```

### Set Configs

```python
CONFIGS = {
    'data_path': '../data/',
    'model_path': '../model/',
    'model_name': 'multi_input',
    
    'valid_start_index': 1704,
    'test_start_index': 1872,
    
    'batch_size': 64,
    'learning_rate': 1e-4,
    'epochs': 100,
    'es_patience': 10,
    
    'window_size': 7*24,
    'target_length': 3,
}
```

### Load Data

```python
train_origin = pd.read_csv(CONFIGS['data_path']+'train.csv', encoding='cp949')

data = deepcopy(train_origin)

data.columns = [
    'num', 'date_time', 'target', 'temp', 'wind',
    'humid', 'rain', 'sun', 'non_elec_eq', 'sunlight_eq'
]

data = data.loc[data['num'] == 1, :]
data
```

![data](/post_images/time_series_3/data.png)

### 시간 데이터 가공

위 데이터의 `date_time`은 있는 그대로 사용할 수 없습니다. 이를 가공해주기 위해 다음 과정을 거친다.

- `str`을 `datetime`으로 형변환 한다.
- `datetime`을 `int`로 변환하여 `timestamp`를 만든다.
- `datetime`에서 연, 월, 일, 요일 정보를 따로 뺀다.
- 순환하는 변수는 cyclical features encoding을 적용한다. ([cyclical features encoding 참고](https://towardsdatascience.com/cyclical-features-encoding-its-about-time-ce23581845ca))
- 공휴일 정보 등 추가로 활용하고 싶은 정보를 추가한다.

```python
def mk_time_data(data):
    
    new_data = data.copy()

    new_data['date_time'] = data['date_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H'))
    
    new_data['time_stamp'] = new_data['date_time'].apply(lambda x: x.timestamp())
    
    new_data['year'] = new_data['date_time'].apply(lambda x: x.year)
    new_data['month'] = new_data['date_time'].apply(lambda x: x.month)
    new_data['day'] = new_data['date_time'].apply(lambda x: x.day)
    
    new_data['hour'] = new_data['date_time'].apply(lambda x: x.hour)
    new_data['cos_hour'] = np.cos(2*np.pi*(new_data['hour']/24))
    new_data['sin_hour'] = np.sin(2*np.pi*(new_data['hour']/24))

    new_data['weekday'] = new_data['date_time'].apply(lambda x: x.weekday())
    new_data['cos_weekday'] = np.cos(2*np.pi*(new_data['weekday']/7))
    new_data['sin_weekday'] = np.sin(2*np.pi*(new_data['weekday']/7))
    
    new_data['is_holiday'] = 0
    new_data.loc[(new_data['weekday'] == 5) | (new_data['weekday'] == 6), 'is_holiday'] = 1
    new_data.loc[(new_data['month'] == 8) & (new_data['day'] == 17), 'is_holiday'] = 1
    
    return new_data


new_data = mk_time_data(data)
new_data
```

![new_data](/post_images/time_series_3/new_data.png)

### Scaling

standard scaling을 진행하며 `inversed_rmse`를 metric으로 활용해주기 위해 `mean`과 `std`는 따로 저장한다.  

```python
def mk_mean_std_dict(data):
    mean_std_dict = {
        col: {
            'mean': data[col].mean(),
            'std': data[col].std()
        } for col in data.columns
    }
    return mean_std_dict


scaling_cols = [
    'temp', 'wind', 'humid', 'rain', 'sun', 'time_stamp', 'target'
]
mean_std_dict = mk_mean_std_dict(new_data[scaling_cols][:CONFIGS['valid_start_index']])
CONFIGS['mean_std_dict'] = mean_std_dict


def standard_scaling(data, mean_std_dict=None):
    if not mean_std_dict:
        mean_std_dict = mk_mean_std_dict(data)
    new_data = data.copy()
    for col in new_data.columns:
        new_data[col] -= mean_std_dict[col]['mean']
        new_data[col] /= mean_std_dict[col]['std']
    return new_data


new_data[scaling_cols] = standard_scaling(new_data[scaling_cols], mean_std_dict)
```

### Dataset

앞서 말했듯이 `temp`, `wind`, `humid` 등과 같은 `target`외의 시계열 데이터 말고도 `target time`의 정보 또한 활용해줄 예정이다.  
`target time`가 평일인지, 주말인지, 공휴일인지, 낮시간인지, 새벽인지 등의 정보가 전력 사용량에 영향을 미칠 것은 당연하다.  
(사실 EDA를 위한 탐색이 선행되어야 하지만, 본 포스팅 시리즈에서는 이 과정은 생략한다.)

현재 `target_length: 3`이기 때문에 맞추고자 하는 3개의 시간 중 가운데 정보만을 활용하도록 하자.  
(이를 그대로 input으로 넣어준다던지, 평균을 이용한다던지 하는 방법도 당연히 가능하다.)

```python
time_series_cols = [
    'temp', 'wind', 'humid', 'rain', 'sun', 'time_stamp',
    'cos_hour', 'sin_hour', 'cos_weekday', 'sin_weekday',
    'is_holiday', 'target',
]
target_time_info_cols = [
    'temp', 'wind', 'humid', 'rain', 'sun', 'time_stamp',
    'cos_hour', 'sin_hour', 'cos_weekday', 'sin_weekday',
    'is_holiday',
]
target_cols = ['target']

CONFIGS['time_series_cols'] = time_series_cols
CONFIGS['target_time_info_cols'] = target_time_info_cols
CONFIGS['target_cols'] = target_cols


def mk_dataset(data, CONFIGS, shuffle=False):
    
    h = CONFIGS['target_length']//2
    
    time_series = data[CONFIGS['time_series_cols']][:-CONFIGS['target_length']]
    target_time_info = data[CONFIGS['target_time_info_cols']]\
        [CONFIGS['window_size']+h:data.shape[0]-CONFIGS['target_length']+1+h]
    target = data[CONFIGS['target_cols']][CONFIGS['window_size']:]
    
    time_series_ds = Dataset.from_tensor_slices(time_series)
    time_series_ds = time_series_ds.window(CONFIGS['window_size'], shift=1, drop_remainder=True)
    time_series_ds = time_series_ds.flat_map(lambda x: x).batch(CONFIGS['window_size'])
    
    target_time_info_ds = Dataset.from_tensor_slices(target_time_info)
    
    target_ds = Dataset.from_tensor_slices(target)
    target_ds = target_ds.window(CONFIGS['target_length'], shift=1, drop_remainder=True)
    target_ds = target_ds.flat_map(lambda x: x).batch(CONFIGS['target_length'])
    
    ds = Dataset.zip(((time_series_ds, target_time_info_ds), target_ds))
    if shuffle:
        ds = ds.shuffle(512)
    ds = ds.batch(CONFIGS['batch_size']).cache().prefetch(2)
    
    return ds


train = new_data.loc[:CONFIGS['valid_start_index'], :]
valid = new_data.loc[CONFIGS['valid_start_index']-CONFIGS['window_size']:CONFIGS['test_start_index'], :]
test = new_data.loc[CONFIGS['test_start_index']-CONFIGS['window_size']:, :]

train_ds = mk_dataset(train, CONFIGS, shuffle=True)
valid_ds = mk_dataset(valid, CONFIGS)
test_ds = mk_dataset(test, CONFIGS)
```

### Modeling

모델 구조를 만들기에 앞서, model metric으로 사용할 inverse된 target을 이용한 `inversed_rmse`를 계산하는 함수를 만든다.

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

사실 지금 만든 custom metric은 약간의 잘못된 점이 있다.  
Metric을 보다 정확하게 customize하는 방법에 대해서는 이어지는 [Multi-Task Learning 시계열 모델 (2)](https://fidabspd.github.io/2022/01/30/time_series_4-2.html)에서 자세히 다뤄보도록 하겠다.

#### Set Model

이제 데이터 하나당 2D-array로 들어오는 시계열 정보를 어떤 모델을 이용하여 훈련할지 결정해야한다.  

이를 활용하는 방법은 정말 다양하다.  
다양한 방법들 중 5가지의 선택지를 만들어보자.
(5가지 외에도 다양한 방법들이 있지만 전부 다룰 수는 없기 때문에 가장 기본적인 내용들로만 구성했다.)

- 1D-array로 바꾼 뒤 `Dense` 활용
- 그대로 `Conv1D` 활용
- 3D-array로 바꾼 뒤 `Conv2D` 활용
- 그대로 `LSTM` 활용
- 그대로 `Bidirectional LSTM` 활용

(각 모델 구조에 대한 자세한 설명은 생략)  

각 모델들의 마지막을 flatten하여 `target time`정보와 `Concatenate`하여 `Dense`레이어를 활용하여 최종 output을 뽑는다.

```python
def set_model(CONFIGS, model_name=None, print_summary=False):
    
    time_series_inputs = Input(batch_shape=(
        None, CONFIGS['window_size'], len(CONFIGS['time_series_cols'])
    ), name='time_series_inputs')
    
    if CONFIGS['model_type'] == 'flatten':
        flatten = Flatten(name='flatten')(time_series_inputs)
    elif CONFIGS['model_type'] == 'cnn1d':
        conv_0 = Conv1D(16, 3, 2, activation='relu', name='conv_0')(time_series_inputs)
        pool_0 = MaxPool1D(2, name='pool_0')(conv_0)
        conv_1 = Conv1D(32, 3, 2, activation='relu', name='conv_1')(pool_0)
        pool_1 = MaxPool1D(2, name='pool_1')(conv_1)
        flatten = Flatten(name='flatten')(pool_1)
    elif CONFIGS['model_type'] == 'cnn2d':
        reshape = Reshape(target_shape=(
            CONFIGS['window_size'], len(CONFIGS['time_series_cols']), 1
        ), name='reshape')(time_series_inputs)
        conv_0 = Conv2D(8, (3, 1), strides=(2, 1), activation='relu', name='conv_0')(reshape)
        pool_0 = MaxPool2D((2, 1), name='pool_0')(conv_0)
        conv_1 = Conv2D(16, (3, 1), strides=(2, 1), activation='relu', name='conv_1')(pool_0)
        pool_1 = MaxPool2D((2, 1), name='pool_1')(conv_1)
        flatten = Flatten(name='flatten')(pool_1)
    elif CONFIGS['model_type'] == 'lstm':
        lstm_0 = LSTM(16, return_sequences=True, activation='relu', name='lstm_0')(time_series_inputs)
        lstm_1 = LSTM(32, activation='relu', name='lstm_1')(lstm_0)
        flatten = Flatten(name='flatten')(lstm_1)
    elif CONFIGS['model_type'] == 'bilstm':
        bilstm_0 = Bidirectional(LSTM(
            16, return_sequences=True, activation='relu'
        ), name='bilstm_0')(time_series_inputs)
        bilstm_1 = Bidirectional(LSTM(
            32, activation='relu'
        ), name='bilstm_1')(bilstm_0)
        flatten = Flatten(name='flatten')(bilstm_1)
        
    target_time_info_inputs = Input(batch_shape=(
        None, len(CONFIGS['target_time_info_cols'])
    ), name='target_time_info_inputs')
    
    concat = Concatenate(name='concat')([flatten, target_time_info_inputs])
        
    dense_0 = Dense(64, activation='relu', name='dense_0')(concat)
    dense_1 = Dense(32, activation='relu', name='dense_1')(dense_0)
    outputs = Dense(CONFIGS['target_length'], name='outputs')(dense_1)
    
    if not model_name:
        model_name = CONFIGS['model_name']
    
    model = Model(
        inputs = [time_series_inputs, target_time_info_inputs],
        outputs = outputs,
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


CONFIGS['model_type'] = 'conv1d'
model = set_model(CONFIGS, print_summary=True)
```

![model_summary](/post_images/time_series_3/model_summary.png)

#### Train

```python
def train_model(model, train_ds, valid_ds, CONFIGS):
    
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
        train_ds,
        batch_size = CONFIGS['batch_size'],
        epochs = CONFIGS['epochs'],
        validation_data = valid_ds,
        callbacks = [
            early_stop,
            save_best_only,
        ]
    )
    
    return history


history = train_model(model, train_ds, valid_ds, CONFIGS)
```

#### Evaluate

```python
best_model = set_model(CONFIGS, model_name='best_'+CONFIGS['model_name'])
best_model.load_weights(f'{CONFIGS["model_path"]}{CONFIGS["model_name"]}.h5')

y_train_pred = best_model.predict(train_ds)
y_valid_pred = best_model.predict(valid_ds)
y_test_pred = best_model.predict(test_ds)

train_loss, train_rmse = best_model.evaluate(train_ds, verbose=0)
valid_loss, valid_rmse = best_model.evaluate(valid_ds, verbose=0)
test_loss, test_rmse = best_model.evaluate(test_ds, verbose=0)

print(f'train_loss: {train_loss:.6f}\ttrain_rmse: {train_rmse:.6f}')
print(f'valid_loss: {valid_loss:.6f}\tvalid_rmse: {valid_rmse:.6f}')
print(f'test_loss: {test_loss:.6f}\ttest_rmse: {test_rmse:.6f}')

# train_loss: 0.028937	train_rmse: 20.117674
# valid_loss: 0.298460	valid_rmse: 67.695808
# test_loss: 0.203631	test_rmse: 55.200130
```

target 시계열 데이터 외에 다른 변수들을 함께 활용하여 multi-input 모델을 완성했다.  

현재까지는 `num == 1`인 데이터만을 사용했다.  
다음 시계열 포스팅 시리즈에서는 60개의 건물을 모두 하나의 모델의 데이터로 활용하는 multi-task learning에 대해 다뤄보자.

## 목차

1. [시계열 데이터의 기본적인 특징과 간단한 모델](https://fidabspd.github.io/2022/01/04/time_series_1.html)
1. [tf.data.Dataset을 이용한 시계열 데이터 구성](https://fidabspd.github.io/2022/01/10/time_series_2.html)
1. [**Multi-Input 시계열 모델**](https://fidabspd.github.io/2022/01/18/time_series_3.html)
1. [Multi-Task Learning 시계열 모델 (1)](https://fidabspd.github.io/2022/01/28/time_series_4-1.html)
1. [Multi-Task Learning 시계열 모델 (2)](https://fidabspd.github.io/2022/01/30/time_series_4-2.html)
1. [시계열 target의 결측](https://fidabspd.github.io/2022/02/10/time_series_5.html)
1. [Recursive Input Prediction](https://fidabspd.github.io/2022/02/17/time_series_6.html)

## 원본 코드 ➞ [<span style="color:#AC1538">CODE (GitHub)</span>](https://github.com/fidabspd/time_series/tree/master/codes/3_multi_input.ipynb)
