---
layout: post
title: Time Series - 시계열 target의 결측
tags: [Time-Series, Tensorflow, Keras]
excerpt_separator: <!--more-->
---

보통의 다른 데이터들은 결측값이 존재한다면 이를 다양한 방법으로 채우게 된다. 시계열 데이터도 마찬가지로 다양한 방법으로 이를 채우면 된다. 하지만 한가지가 크게 다르다.  
그것은 바로 독립변수를 곧 종속변수로 활용하기 때문이고, 이 작은 차이가 굉장히 큰 차이를 만든다.  
<!--more-->
독립변수에 결측값이 존재한다면 어떻게든 채워넣으면 그만이지만 종속변수에 결측이 존재한다면 어떻게 해야할까? 이를 자세히 알아보도록 하자.

## 목차

1. [시계열 데이터의 기본적인 특징과 간단한 모델](https://fidabspd.github.io/2022/01/04/time_series_1.html)
1. [tf.data.Dataset을 이용한 시계열 데이터 구성](https://fidabspd.github.io/2022/01/10/time_series_2.html)
1. [Multi-Input 시계열 모델](https://fidabspd.github.io/2022/01/18/time_series_3.html)
1. [Multi-Task Learning 시계열 모델 (1)](https://fidabspd.github.io/2022/01/28/time_series_4-1.html)
1. [Multi-Task Learning 시계열 모델 (2)](https://fidabspd.github.io/2022/01/30/time_series_4-2.html)
1. [**시계열 target의 결측**](https://fidabspd.github.io/2022/02/10/time_series_5.html)
1. [Recursive Input Prediction](https://fidabspd.github.io/2022/02/17/time_series_6.html)

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

## 일반적인 결측값

일반적으로 '데이터에 결측값이 존재한다'는 의미는 독립변수에 빈 값이 있다는 의미이다. 이는 데이터에 따라 굉장히 다양한 방법을 사용해 채울 수 있다.  
시계열 데이터도 마찬가지이다. 독립변수들에 결측값이 존재한다면 다양한 방법으로 채우면 그만이다. 하지만 시계열 데이터는 현재의 target값을 이후에 input으로 사용한다는 특징이 존재한다. 따라서 `target`(현재는 `전력사용량`)에 결측값이 존재한다면 이는 곧 독립변수의 결측을 의미함과 동시에, 예측해야하는 종속변수의 결측을 의미한다.  

그렇다면 보통 종속변수에 결측값이 있다면 어떻게 할까. 독립변수의 결측값을 채우듯이 종속변수의 결측값을 채운다고 가정해보자.  
이 데이터는 정답이 아니다. 어디까지나 예측해서 채워넣은 `y_hat`이다. 이를 정답으로 생각하고 훈련을 시킨다면 둘 중 하나의 경우인데.  

1. 결측값을 대체한 값들이 정말로 정답이다. → 모델을 훈련할 이유가 없다. 이미 정답을 어떻게 채우는지 알고 있다.
2. 결측값을 대체한 값들이 단순히 예측하여 채워넣은 값에 불과하며 정답이 아니다. → 모델은 정답이 아닌 값들로 학습을 하고있다.

당연히 실제의 경우 2번의 경우가 대부분이다.  
결론은 보다시피 종속변수의 결측값을 대체하여 채울 경우 좋은 결과로 이어지지 않는다. 따라서 대부분의 경우 예측해야하는 target 값이 결측값일 경우 해당 데이터 자체를 삭제하는 경우가 대부분이다.

그렇다면 시계열 데이터 역시도 target 값에 결측값이 존재한다면 해당 데이터를 지워버리면 되는거 아닌가? 하는 의문이 생긴다.  
물론 맞는말이다. input으로 들어가는 시계열 데이터는 적당히 결측값을 대체하여 사용하고 target으로 사용될 경우 해당 데이터를 삭제하면 된다.  
하지만 본 시계열 시리즈에서 `target_length = 3`으로 유지해왔다. 따라서 target 하나의 결측은 곧 데이터 3개를 지워야하는 상황에 이르게 된다.  
데이터를 최대한 지우지 않고 유지하면서 학습을 하려면 어떻게 해야할까?

의외로 간단하다. 결측값을 대체한 `target`의 경우에는 해당 데이터로 backpropagation을 진행하지 않으면 된다. 이를 코드로 구현해보자.

## CODE

코드는 바로 전 시계열 시리즈인 [Multi-Task Learning 시계열 모델](https://fidabspd.github.io/2022/01/28/time_series_4-1.html)과 거의 유사하다.

### Libraries

```python
from copy import deepcopy

import numpy as np
import pandas as pd

import datetime

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
```

### Set Configs

```python
CONFIGS = {
    'data_path': '../data/',
    'model_path': '../model/',
    'model_name': 'handling_missing_value',
    'model_type': 'cnn1d',
    
    'dtype': tf.float32,
    
    'valid_start_date_time': '2020-08-11 00',
    'test_start_date_time': '2020-08-18 00',
    
    'batch_size': 64,
    'learning_rate': 5e-5,
    'epochs': 100,
    'es_patience': 10,
    
    'window_size': 7*24,
    'shift': 1,
    'target_length': 3,
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

CONFIGS['n_buildings'] = len(data['num'].unique())
```

### 결측값 추가

```python
missing_values = np.random.choice(data.shape[0], int(data.shape[0]*0.3), replace=False)
data.loc[missing_values, 'target'] = np.nan
```

실제 데이터에는 결측값이 없기 때문에 임의로 30%의 결측값을 `target`에 추가한다.

### 시간 데이터 가공

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
```

### Make Building Info

```python
def mk_building_info(data, data_for_calc, CONFIGS):
        
    new_data = data.copy()
    new_data['range'] = 0
    new_data['mean'] = 0
    new_data['std'] = 0
    new_data['holiday_gap'] = 0
    new_data['day_gap'] = 0

    for num in range(CONFIGS['n_buildings']):
        building = data_for_calc.query(f'num == {num}')
        
        bt_range = building['target'].max()-building['target'].min()
        bt_mean = building['target'].mean()
        bt_std = building['target'].std()
        bt_holiday_gap = abs(building.query('is_holiday == 0')['target'].mean() - building.query('is_holiday == 1')['target'].mean())
        bt_day_gap = 0
        for d in range(building.shape[0]//24):
            tmp = building['target'][d*24:(d+1)*24]
            bt_day_gap += (tmp.max()-tmp.min())/(building.shape[0]//24)
            
        new_data.loc[new_data['num']==num, 'range'] = bt_range
        new_data.loc[new_data['num']==num, 'mean'] = bt_mean
        new_data.loc[new_data['num']==num, 'std'] = bt_std
        new_data.loc[new_data['num']==num, 'holiday_gap'] = bt_holiday_gap
        new_data.loc[new_data['num']==num, 'day_gap'] = bt_day_gap
        
    new_data['mean_to_inverse'] = new_data['mean']
    new_data['std_to_inverse'] = new_data['std']
        
    return new_data


new_data = mk_building_info(
    new_data,
    new_data[new_data['date_time']<CONFIGS['valid_start_date_time']],
    CONFIGS
)

new_data
```

### Scaling

```python
def mk_mean_std_dict(data, scaling_by_building_cols):
    mean_std_dict = {}
    for num in range(60):
        building = data.query(f'num == {num}')
        mean_std_dict[num] = {
            col: {
                'mean': building[col].mean(),
                'std': building[col].std()
            } for col in scaling_by_building_cols
        }
    return mean_std_dict


scaling_by_building_cols = [
    'temp', 'wind', 'humid', 'rain', 'sun', 'time_stamp', 'target',
]
scaling_by_all_cols = ['range', 'mean', 'std', 'holiday_gap', 'day_gap']

mean_std_dict = mk_mean_std_dict(
    new_data[new_data['date_time'] < CONFIGS['valid_start_date_time']],
    scaling_by_building_cols
)
CONFIGS['mean_std_dict'] = mean_std_dict


def standard_scaling(data, scaling_by_building_cols, scaling_by_all_cols, mean_std_dict=None):
    if not mean_std_dict:
        mean_std_dict = mk_mean_std_dict(data, scaling_by_building_cols)
        
    new_data = data.copy()
    for num in range(60):
        for col in scaling_by_building_cols:
            new_data.loc[new_data['num']==num, col] -= mean_std_dict[num][col]['mean']
            new_data.loc[new_data['num']==num, col] /= mean_std_dict[num][col]['std']
    
    for col in scaling_by_all_cols:
        m = new_data.loc[:, col].mean()
        s = new_data.loc[:, col].std()
        new_data.loc[:, col] -= m
        new_data.loc[:, col] /= s
    
    return new_data


new_data = standard_scaling(new_data, scaling_by_building_cols, scaling_by_all_cols, mean_std_dict)
```

scaling은 결측값을 채우기 이전의 값으로 진행한다.

### Visualize

결측값을 어떻게 채울 지 결정하기 위해 train에 사용할 데이터만을 이용하여 간단한 시각화를 진행한다.  
요일별, 시간별, 일별 `target`의 평균을 시각화 한다.

```python
def visualize(data):

    eda_df = data[(data['date_time']<CONFIGS['valid_start_date_time'])]

    weekday_mean = eda_df.groupby(
        ['num', 'weekday'], as_index=False)['target'].mean().\
        pivot('num', 'weekday', 'target')

    hour_mean = eda_df.groupby(
        ['num', 'hour'], as_index=False)['target'].mean().\
        pivot('num', 'hour', 'target')

    day_mean = eda_df.groupby(
        ['num', 'day'], as_index=False)[['target']].mean().\
        pivot('num', 'day', 'target')


    fig = plt.figure(figsize = (20,4))
    for i in range(weekday_mean.shape[0]):
        plt.plot(weekday_mean.iloc[i, :], alpha = 0.5, linewidth = 0.5)
    plt.plot(weekday_mean.mean(axis=0), color='b', alpha = 0.7, linewidth=2)
    plt.xticks(range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.title('Weekday mean')

    fig = plt.figure(figsize = (20,4))
    for i in range(hour_mean.shape[0]):
        plt.plot(hour_mean.iloc[i, :], alpha = 0.5, linewidth = 0.5)
    plt.plot(hour_mean.mean(axis=0), color='b', alpha = 0.7, linewidth=2)
    plt.xticks(range(24))
    plt.title('Hour mean')

    fig = plt.figure(figsize = (20,4))
    for i in range(day_mean.shape[0]):
        plt.plot(day_mean.iloc[i, :], alpha = 0.5, linewidth = 0.5)
    plt.plot(day_mean.mean(axis=0), color='b', alpha = 0.7, linewidth=2)
    plt.xticks(range(1, 32))
    plt.title('Day mean')

    plt.show()


visualize(new_data)
```

![target_visual](/assets/img/posts/time_series_5/target_visual.png)

위 데이터에서 볼 수 있듯 대부분의 시계열 데이터는 요일, 시간, 날짜 등의 변수들에 굉장히 의존적이다.  
현재는 multi-task learning을 진행중이므로 건물별 특징 또한 포함시켜야한다.  
따라서 단순히 column의 평균을 이용한 결측값 대체는 곧 이상값을 집어넣는 격이 되는 경우가 많다.  

결측값이 어떻게 분포하는지도 살펴보도록 하자.

```python
def visualize_na_rate(data):
    eda_df = data.copy()

    eda_df['is_holiday'] = eda_df['is_holiday'].astype('category')
    cols = ['num', 'day', 'hour', 'weekday', 'is_holiday']

    fig = plt.figure(figsize = (len(cols)*5.7, 1*5))

    eda_df['is_na'] = eda_df['target'].isna().apply(int)
    for idx, col in enumerate(cols):
        tmp = eda_df.groupby([col], as_index=False)['is_na'].mean()
        plt.subplot(1, len(cols), idx+1)
        plt.bar(tmp[col], tmp['is_na'])
        plt.title(col)

    
visualize_na_rate(new_data)
```

![na_dist](/assets/img/posts/time_series_5/na_dist.png)

그냥 무작위로 결측값을 만들었기 때문에 건물번호, 날짜, 시간, 요일, 휴일여부 등에 관계 없이 고르게 결측값이 분포하는 모습이 보인다.  
만약 그렇지 않고 특정한 곳에만 결측값이 몰려있다면 해당 결측값을 더 잘 채우기위해 고민해야한다.  
간단하게 예를들어 00시~06시의 데이터에만 결측값이 연속한 형태로 몰려있다면 단순한 선형 interpolation은 적절하지 않은 결측값 대체이다.

### Fill Missing Value

본 포스팅에서는 결측값이 고르게 분포해있기 때문에 선형 interpolation([wikipedia 참고](https://en.wikipedia.org/wiki/Linear_interpolation))을 진행하고, 각 건물의 `target` 시퀀스의 시작에 결측이 있다면 가장 가까운 실제 값으로 채우도록 한다.  
또한 해당 `target`값이 진짜 true값인지, 선형 보간된 값인지를 구분하는 column을 데이터에 추가한다.
(실제 결측값을 대체할 때는 보다 깊은 데이터에 대한 이해를 바탕으로 진행해야한다.)

```python
def fill_missing_value(data):
    new_data = data.copy()
    new_data['is_true'] = new_data['target'].notna().apply(int)
    for i in range(60):
        building = new_data[new_data['num'] == i]
        new_data.loc[new_data['num'] == i, 'target'] = building['target'].interpolate()
        fill = new_data.loc[(new_data['num'] == i)&(new_data['target'].notna()), 'target'].tolist()[0]
        new_data.loc[(new_data['num'] == i)&(new_data['target'].isna()), 'target'] = fill
    return new_data


new_data = fill_missing_value(new_data)
```

### Dataset

#### Input Columns

결측값이 참인지 거짓인지 구분하기 위한 column인 `is_true`도 포함한다.  
이는 모델의 독립변수로도 활용할 예정이며, 본 포스팅의 시작에서 말했던 loss를 계산한 이후 backpropagation을 진행할지 여부를 결정하는 데에도 사용할 예정이다.

```python
building_num_cols = ['num']
building_info_cols = [
    'range', 'mean', 'std', 'holiday_gap', 'day_gap',
    'non_elec_eq', 'sunlight_eq',
]
target_time_info_cols = [
    'temp', 'wind', 'humid', 'rain', 'sun', 'time_stamp',
    'cos_hour', 'sin_hour', 'cos_weekday', 'sin_weekday',
    'is_holiday',
]
time_series_cols = [
    'temp', 'wind', 'humid', 'rain', 'sun', 'time_stamp',
    'cos_hour', 'sin_hour', 'cos_weekday', 'sin_weekday',
    'is_holiday', 'is_true', 'target',
]
to_inverse_cols = ['mean_to_inverse', 'std_to_inverse']
is_true_cols = ['is_true']
target_cols = ['target']
input_cols = list(set(
    building_num_cols + building_info_cols + target_time_info_cols +
    time_series_cols + target_cols + to_inverse_cols + is_true_cols
))


CONFIGS['building_num_cols'] = building_num_cols
CONFIGS['building_info_cols'] = building_info_cols
CONFIGS['target_time_info_cols'] = target_time_info_cols
CONFIGS['time_series_cols'] = time_series_cols
CONFIGS['to_inverse_cols'] = to_inverse_cols
CONFIGS['is_true_cols'] = is_true_cols
CONFIGS['input_cols'] = input_cols
CONFIGS['target_cols'] = target_cols
```

#### Make tf.data.Dataset

```python
def crop(data, crop_type, CONFIGS):
    building_length = tf.shape(data)[0]
    if crop_type == 'one_row':
        h = CONFIGS['target_length']//2
        croped = data[CONFIGS['window_size']+h:building_length-CONFIGS['target_length']+1+h]
    elif crop_type == 'time_series_input':
        croped = data[:-CONFIGS['target_length']]
    elif crop_type == 'target':
        croped = data[CONFIGS['window_size']:]
    return croped


def mk_window(data, size, shift):
    ds = Dataset.from_tensor_slices(data)
    ds = ds.window(
        size=size, shift=shift, drop_remainder=True
    ).flat_map(lambda x: x).batch(size)
    return ds


def mk_time_series(data, building_length, crop_type, CONFIGS, dtype=None):
    if not dtype:
        dtype = CONFIGS['dtype']

    ds = Dataset.from_tensor_slices(data).batch(building_length)
    if crop_type == 'one_row':
        ds = ds.map(lambda x: crop(x, crop_type, CONFIGS))
        ds = ds.flat_map(lambda x: Dataset.from_tensor_slices(x))
    elif crop_type == 'time_series_input':
        ds = ds.map(lambda x: crop(x, crop_type, CONFIGS))
        ds = ds.flat_map(
            lambda x: mk_window(x, CONFIGS['window_size'], CONFIGS['shift']))
    elif crop_type == 'target':
        ds = ds.map(lambda x: crop(x, crop_type, CONFIGS))
        ds = ds.flat_map(
            lambda x: mk_window(x, CONFIGS['target_length'], CONFIGS['shift']))
    ds.map(lambda x: tf.cast(x, dtype))

    return ds


def mk_dataset(data, CONFIGS, shuffle=False):

    data = data[CONFIGS['input_cols']]
    building_length = data.shape[0]//60

    building_num = data[CONFIGS['building_num_cols']]
    building_info = data[CONFIGS['building_info_cols']]
    target_time_info = data[CONFIGS['target_time_info_cols']]
    time_series = data[CONFIGS['time_series_cols']]
    to_inverse = data[CONFIGS['to_inverse_cols']]
    is_true = data[CONFIGS['is_true_cols']]
    target = data[CONFIGS['target_cols']]

    building_num_ds = mk_time_series(building_num, building_length, 'one_row', CONFIGS, tf.int16)
    building_info_ds = mk_time_series(building_info, building_length, 'one_row', CONFIGS)
    target_time_info_ds = mk_time_series(target_time_info, building_length, 'one_row', CONFIGS)
    time_series_ds = mk_time_series(time_series, building_length, 'time_series_input', CONFIGS)
    to_inverse_ds = mk_time_series(to_inverse, building_length, 'target', CONFIGS)
    is_true_ds = mk_time_series(is_true, building_length, 'target', CONFIGS)
    target_ds = mk_time_series(target, building_length, 'target', CONFIGS)
    
    # zip
    ds = Dataset.zip((
        (
            building_num_ds,
            building_info_ds,
            target_time_info_ds,
            time_series_ds,
            to_inverse_ds,
            is_true_ds
        ),
        target_ds
    ))
    if shuffle:
        ds = ds.shuffle(512)
    ds = ds.batch(CONFIGS['batch_size']).cache().prefetch(2)
    
    return ds


str_to_dt = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H')
hour_to_td = lambda x: datetime.timedelta(hours=x)

train = new_data.loc[
    new_data['date_time'] < \
        str_to_dt(CONFIGS['valid_start_date_time']),
    :
]
valid = new_data.loc[
    (new_data['date_time'] >= \
        str_to_dt(CONFIGS['valid_start_date_time'])-hour_to_td(CONFIGS['window_size']))&\
    (new_data['date_time'] < \
         str_to_dt(CONFIGS['test_start_date_time'])),
    :
]
test = new_data.loc[
    new_data['date_time'] >= \
        str_to_dt(CONFIGS['test_start_date_time'])-hour_to_td(CONFIGS['window_size']),
    :
]

train_ds = mk_dataset(train, CONFIGS, shuffle=True)
valid_ds = mk_dataset(valid, CONFIGS)
test_ds = mk_dataset(test, CONFIGS)
```

### Modeling

### Customize Loss & Metric

본 포스팅의 시작에서 말했던 loss를 계산한 이후 backpropagation을 진행할지 여부를 결정하는, 본 포스팅의 가장 핵심이 되는 부분이다.  
`Loss`와 `Metric`을 custom하는 방법은 이전 포스팅에서 다뤘으므로 사실 어렵진 않다.  
각 데이터에서 계산된 loss에 `is_true`를 곱해주는 방식으로 간단하게 구현이 가능하다.  

만약 `target`이 실제 데이터가 아니라 선형보간된 값이라면 loss는 0이되어 해당 데이터로는 backpropagation을 진행하지 않는다.

```python
class CustomMSE(Loss):
    
    def __init__(self, target_max, name="custom_mse"):
        super(CustomMSE, self).__init__(name=name)
        self.target_max = target_max

    def call(self, y_true, y_pred):
        y_true = tf.squeeze(y_true)
        y_pred, mean, std, is_true = \
            y_pred[..., 0], y_pred[..., 1], y_pred[..., 2], y_pred[..., 3]

        y_true_inversed = y_true*std+mean
        y_pred_inversed = y_pred*std+mean
        
        y_true_inversed_scaled = y_true_inversed/self.target_max
        y_pred_inversed_scaled = y_pred_inversed/self.target_max

        mse = tf.reduce_mean((y_true_inversed_scaled-y_pred_inversed_scaled)**2)
        return mse
```

`Metric`도 마찬가지로 선형보간 된 `target`을 잘 맞췄는지는 중요하지 않으므로 해당 데이터를 이용한 metirc역시 0으로 바꿔준다.

```python
class InversedRMSE(Metric):
    
    def __init__(self, CONFIGS, name="inversed_rmse", **kwargs):
        super(InversedRMSE, self).__init__(name=name, **kwargs)
        self.inversed_mse = self.add_weight(name='inversed_mse', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.CONFIGS = CONFIGS

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(y_true)
        y_pred, mean, std, is_true = \
            y_pred[..., 0], y_pred[..., 1], y_pred[..., 2], y_pred[..., 3]

        y_true_inversed = y_true*std+mean
        y_pred_inversed = y_pred*std+mean

        error = tf.reduce_sum(tf.math.squared_difference(y_true_inversed, y_pred_inversed))
        
        self.inversed_mse.assign_add(error)
        self.count.assign_add(tf.cast(tf.size(y_true), CONFIGS['dtype']))

    def result(self):
        return tf.sqrt(tf.math.divide_no_nan(self.inversed_mse, self.count))
```

#### Customize Layer

```python
class BuildingNum(Layer):

    def __init__(self, CONFIGS, name='building_num_layer', **kwargs):
        super(BuildingNum, self).__init__(name=name, **kwargs)
        self.building_num_emb = Embedding(
            input_dim=CONFIGS['n_buildings'],
            output_dim=CONFIGS['embedding_dim']
        )
        self.bn_outputs = Reshape(target_shape=(CONFIGS['embedding_dim'],))
        
    def get_config(self):
        config = super(BuildingNum, self).get_config().copy()
        config.update({
            'building_num_emb': self.building_num_emb,
            'bn_outputs': self.bn_outputs,
        })
        return config
        
    def call(self, inputs):
        x = self.building_num_emb(inputs)
        outputs = self.bn_outputs(x)
        return outputs
    

class BuildingInfo(Layer):
    
    def __init__(self, CONFIGS, name='building_info_layer', **kwargs):
        super(BuildingInfo, self).__init__(name=name, **kwargs)
        self.bi_dense_0 = Dense(16, activation='relu')
        self.dropout_0 = Dropout(0.3)
        self.bi_outputs = Dense(32, activation='relu')
        
    def get_config(self):
        config = super(BuildingInfo, self).get_config().copy()
        config.update({
            'bi_dense_0': self.bi_dense_0,
            'dropout_0': self.dropout_0,
            'bi_outputs': self.bi_outputs,
        })
        return config
        
    def call(self, inputs):
        x = self.bi_dense_0(inputs)
        x = self.dropout_0(x)
        outputs = self.bi_outputs(x)
        return outputs
    

class TargetTimeInfo(Layer):
    
    def __init__(self, CONFIGS, name='target_time_info_layer', **kwargs):
        super(TargetTimeInfo, self).__init__(name=name, **kwargs)
        self.tti_dense_0 = Dense(16, activation='relu')
        self.dropout_0 = Dropout(0.3)
        self.tti_outputs = Dense(32, activation='relu')
        
    def get_config(self):
        config = super(TargetTimeInfo, self).get_config().copy()
        config.update({
            'tti_dense_0': self.tti_dense_0,
            'dropout_0': self.dropout_0,
            'tti_outputs': self.tti_outputs,
        })
        return config
        
    def call(self, inputs):
        x = self.tti_dense_0(inputs)
        x = self.dropout_0(x)
        outputs = self.tti_outputs(x)
        return outputs
    

class TimeSeries(Layer):
    
    def __init__(self, CONFIGS, name='time_series_layer', **kwargs):
        super(TimeSeries, self).__init__(name=name, **kwargs)
        
        if CONFIGS['model_type'] == 'flatten':
            pass
        elif CONFIGS['model_type'] == 'cnn1d':
            self.conv1d_0 = Conv1D(16, 3, 2, activation='relu')
            self.pool1d_0 = MaxPool1D(2)
            self.conv1d_1 = Conv1D(32, 3, 2, activation='relu')
            self.pool1d_1 = MaxPool1D(2)
        elif CONFIGS['model_type'] == 'cnn2d':
            self.conv2d_reshape = Reshape(target_shape=(
                CONFIGS['window_size'], len(CONFIGS['time_series_cols']), 1
            ))
            self.conv2d_0 = Conv2D(8, (3, 1), strides=(2, 1), activation='relu')
            self.pool2d_0 = MaxPool2D((2, 1))
            self.conv2d_1 = Conv2D(16, (3, 1), strides=(2, 1), activation='relu')
            self.pool2d_1 = MaxPool2D((2, 1))
        elif CONFIGS['model_type'] == 'lstm':
            self.lstm_0 = LSTM(16, return_sequences=True, activation='relu')
            self.lstm_1 = LSTM(32, activation='relu')
        elif CONFIGS['model_type'] == 'bilstm':
            self.bilstm_0 = Bidirectional(LSTM(16, return_sequences=True, activation='relu'))
            self.bilstm_1 = Bidirectional(LSTM(32, activation='relu'))
        self.time_series_outputs = Flatten()
        
    def get_config(self):
        config = super(TimeSeries, self).get_config().copy()
        if CONFIGS['model_type'] == 'flatten':
            pass
        elif CONFIGS['model_type'] == 'cnn1d':
            config.update({
                'conv1d_0': self.conv1d_0,
                'pool1d_0': self.pool1d_0,
                'conv1d_1': self.conv1d_1,
                'pool1d_1': self.pool1d_1,
            })
        elif CONFIGS['model_type'] == 'cnn2d':
            config.update({
                'conv2d_reshape': self.conv2d_reshape,
                'conv2d_0': self.conv2d_0,
                'pool2d_0': self.pool2d_0,
                'conv2d_1': self.conv2d_1,
                'pool2d_1': self.pool2d_1,
            })
        elif CONFIGS['model_type'] == 'lstm':
            config.update({
                'lstm_0': self.lstm_0,
                'lstm_1': self.lstm_1,
            })
        elif CONFIGS['model_type'] == 'bilstm':
            config.update({
                'bilstm_0': self.bilstm_0,
                'bilstm_1': self.bilstm_1,
            })
        config.update({
            'time_series_outputs': self.time_series_outputs,
        })
        return config
        
    def call(self, inputs):
        if CONFIGS['model_type'] == 'flatten':
            x = inputs
        elif CONFIGS['model_type'] == 'cnn1d':
            x = self.conv1d_0(inputs)
            x = self.pool1d_0(x)
            x = self.conv1d_1(x)
            x = self.pool1d_1(x)
        elif CONFIGS['model_type'] == 'cnn2d':
            x = self.conv2d_reshape(x)
            x = self.conv2d_0(x)
            x = self.pool2d_0(x)
            x = self.conv2d_1(x)
            x = self.pool2d_1(x)
        elif CONFIGS['model_type'] == 'lstm':
            x = self.lstm_0(x)
            x = self.lstm_1(x)
        elif CONFIGS['model_type'] == 'bilstm':
            x = self.bilstm_0(x)
            x = self.bilstm_1(x)
        outputs = self.time_series_outputs(x)
        return outputs
```

#### Set Model

```python
def set_model(CONFIGS, model_name=None, print_summary=False):
    
    # building_num
    building_num_inputs = Input(batch_shape=(None, 1), name='building_num_inputs')
    bn_outputs = BuildingNum(CONFIGS)(building_num_inputs)
    
    # building_info
    building_info_inputs = Input(
        batch_shape=(None, len(CONFIGS['building_info_cols'])),
        name='building_info_inputs'
    )
    bi_outputs = BuildingInfo(CONFIGS)(building_info_inputs)
    
    # target_time_info
    target_time_info_inputs = Input(
        batch_shape=(None, len(CONFIGS['target_time_info_cols'])),
        name='target_time_info_inputs'
    )
    tti_outputs = TargetTimeInfo(CONFIGS)(target_time_info_inputs)
    
    # time_series
    time_series_inputs = Input(batch_shape=(
        None, CONFIGS['window_size'], len(CONFIGS['time_series_cols'])
    ), name='time_series_inputs')
    time_series_outputs = TimeSeries(CONFIGS)(time_series_inputs)
    
    concat = Concatenate(name='concat')([bn_outputs, bi_outputs, tti_outputs, time_series_outputs])
        
    dense_0 = Dense(64, activation='relu', name='dense_0')(concat)
    dropout_0 = Dropout(0.3, name='dropout_0')(dense_0)
    dense_1 = Dense(32, activation='relu', name='dense_1')(dropout_0)
    dropout_1 = Dropout(0.3, name='dropout_0')(dense_1)
    dense_2 = Dense(CONFIGS['target_length'], name='dense_2')(dense_1)
    
    reshape_dense = Reshape(target_shape=(
        CONFIGS['target_length'], 1
    ), name='reshape_dense')(dense_2)
    
    # to_inverse & is_true
    to_inverse_inputs = Input(batch_shape=(
        None, CONFIGS['target_length'], len(CONFIGS['to_inverse_cols'])
    ), name='to_inverse_inputs')
    is_true_inputs = Input(batch_shape=(
        None, CONFIGS['target_length'], len(CONFIGS['is_true_cols'])
    ), name='is_true_inputs')
    outputs = Concatenate(axis=-1, name='concat_to_inverse')([
        reshape_dense, to_inverse_inputs, is_true_inputs
    ])
    
    if not model_name:
        model_name = CONFIGS['model_name']
    
    model = Model(
        inputs = [
            building_num_inputs,
            building_info_inputs,
            target_time_info_inputs,
            time_series_inputs,
            to_inverse_inputs,
            is_true_inputs,
        ],
        outputs = outputs,
        name = model_name
    )
    
    custom_mse = CustomMSE(CONFIGS['target_max'])
    inversed_rmse = InversedRMSE(CONFIGS)
    optimizer = Adam(learning_rate=CONFIGS['learning_rate'])
    model.compile(
        loss = custom_mse,
        optimizer = optimizer,
        metrics = [inversed_rmse],
    )
    
    if print_summary:
        model.summary()
    
    return model


CONFIGS['target_max'] = \
    data[data['date_time']<CONFIGS['valid_start_date_time']]['target'].max()
CONFIGS['embedding_dim'] = 10

model = set_model(CONFIGS, print_summary=True)
```

```
Model: "handling_missing_value"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 building_num_inputs (InputLaye  [(None, 1)]         0           []                               
 r)                                                                                               
                                                                                                  
 building_info_inputs (InputLay  [(None, 7)]         0           []                               
 er)                                                                                              
                                                                                                  
 target_time_info_inputs (Input  [(None, 11)]        0           []                               
 Layer)                                                                                           
                                                                                                  
 time_series_inputs (InputLayer  [(None, 168, 13)]   0           []                               
 )                                                                                                
                                                                                                  
 building_num_layer (BuildingNu  (None, 10)          600         ['building_num_inputs[0][0]']    
 m)                                                                                               
                                                                                                  
 building_info_layer (BuildingI  (None, 32)          672         ['building_info_inputs[0][0]']   
 nfo)                                                                                             
                                                                                                  
 target_time_info_layer (Target  (None, 32)          736         ['target_time_info_inputs[0][0]']
 TimeInfo)                                                                                        
                                                                                                  
 time_series_layer (TimeSeries)  (None, 320)         2208        ['time_series_inputs[0][0]']     
                                                                                                  
 concat (Concatenate)           (None, 394)          0           ['building_num_layer[0][0]',     
                                                                  'building_info_layer[0][0]',    
                                                                  'target_time_info_layer[0][0]', 
                                                                  'time_series_layer[0][0]']      
                                                                                                  
 dense_0 (Dense)                (None, 64)           25280       ['concat[0][0]']                 
                                                                                                  
 dropout_0 (Dropout)            (None, 64)           0           ['dense_0[0][0]']                
                                                                                                  
 dense_1 (Dense)                (None, 32)           2080        ['dropout_0[0][0]']              
                                                                                                  
 dense_2 (Dense)                (None, 3)            99          ['dense_1[0][0]']                
                                                                                                  
 reshape_dense (Reshape)        (None, 3, 1)         0           ['dense_2[0][0]']                
                                                                                                  
 to_inverse_inputs (InputLayer)  [(None, 3, 2)]      0           []                               
                                                                                                  
 is_true_inputs (InputLayer)    [(None, 3, 1)]       0           []                               
                                                                                                  
 concat_to_inverse (Concatenate  (None, 3, 4)        0           ['reshape_dense[0][0]',          
 )                                                                'to_inverse_inputs[0][0]',      
                                                                  'is_true_inputs[0][0]']         
                                                                                                  
==================================================================================================
Total params: 31,675
Trainable params: 31,675
Non-trainable params: 0
__________________________________________________________________________________________________
```

최종 `Dense`레이어를 지나고난 뒤, scaling을 inverse해주기 위한 `to_inverse_inputs`와 target 값이 선형 보간된 값인지 판단하기 위한 `is_true_inputs`를 `Concatenate`해준다.  
모델의 최종 output의 shape은 `(None, 3, 4)`가 된다. (`(batch_size, target_length, [pred, mean, std, is_true])`)

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
    tensorboard_callback = TensorBoard(
        log_dir = CONFIGS['tensorboard_log_path']
    )
    
    history = model.fit(
        train_ds,
        batch_size = CONFIGS['batch_size'],
        epochs = CONFIGS['epochs'],
        validation_data = valid_ds,
        callbacks = [
            early_stop,
            save_best_only,
            tensorboard_callback,
        ]
    )
    
    return history


history = train_model(model, train_ds, valid_ds, CONFIGS)
```

#### Evaluate

```python
best_model = set_model(CONFIGS, model_name='best_'+CONFIGS['model_name'])
best_model.load_weights(f'{CONFIGS["model_path"]}{CONFIGS["model_name"]}.h5')

train_loss, train_rmse = best_model.evaluate(train_ds, verbose=0)
valid_loss, valid_rmse = best_model.evaluate(valid_ds, verbose=0)
test_loss, test_rmse = best_model.evaluate(test_ds, verbose=0)

print(f'train_loss: {train_loss:.6f}\ttrain_rmse: {train_rmse:.6f}')
print(f'valid_loss: {valid_loss:.6f}\tvalid_rmse: {valid_rmse:.6f}')
print(f'test_loss: {test_loss:.6f}\ttest_rmse: {test_rmse:.6f}')
```

```
train_loss: 0.000177	train_rmse: 224.546661
valid_loss: 0.000667	valid_rmse: 435.688721
test_loss: 0.000498	test_rmse: 376.542938
```

이렇게 결측값이 있는 시계열 데이터를 이용하여 1보다 큰 `target_length`를 가지는 모델을 완성하였다. 

양이 좀 많지만 반복된 내용이 거의 대부분이라 하나의 포스팅으로 구성했다.  
다음 포스팅에서는 `target`의 예측값을 recursive하게 다시 `input`으로 집어넣는 방법에 대해 다뤄보자.

## 목차

1. [시계열 데이터의 기본적인 특징과 간단한 모델](https://fidabspd.github.io/2022/01/04/time_series_1.html)
1. [tf.data.Dataset을 이용한 시계열 데이터 구성](https://fidabspd.github.io/2022/01/10/time_series_2.html)
1. [Multi-Input 시계열 모델](https://fidabspd.github.io/2022/01/18/time_series_3.html)
1. [Multi-Task Learning 시계열 모델 (1)](https://fidabspd.github.io/2022/01/28/time_series_4-1.html)
1. [Multi-Task Learning 시계열 모델 (2)](https://fidabspd.github.io/2022/01/30/time_series_4-2.html)
1. [**시계열 target의 결측**](https://fidabspd.github.io/2022/02/10/time_series_5.html)
1. [Recursive Input Prediction](https://fidabspd.github.io/2022/02/17/time_series_6.html)

## 원본 코드 ➞ [<span style="color:#AC1538">CODE (GitHub)</span>](https://github.com/fidabspd/time_series/blob/master/codes/5_handling_missing_value.ipynb)
