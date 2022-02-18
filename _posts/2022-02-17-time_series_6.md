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
(앞으로 2번 방법을 **rolling prediction**이라고 부르도록 한다. 완벽하게 정의된 단어는 아니지만 이렇게 지칭하면 대부분 의사소통에는 문제가 없다.)

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
그 이유는 첫번째 예측이 잘못되면 두번째 예측부터는 아예 잘못된 값이 input으로 들어가게 되며 이는 연쇄적인 효과를 일으켜 sequence가 길어질수록 왜곡이 커지기 때문이다.  
하지만 본 포스팅의 목적은 시계열 데이터를 이용하여 모델을 만드는 과정에서 마주하는 다양한 어려움들을 어떻게 해결하면 좋을지에 대해 정리하는 내용이기 때문에 **1번** 방법보다는 더 어려운 방법인 **2번** 방법을 써보도록 하자.

## CODE

[tf.data.Dataset 구성](#dataset) 이전 코드는 [Multi-Task Learning 시계열 모델](https://fidabspd.github.io/2022/01/28/time_series_4-1.html)과 거의 유사하다.  
따라서 설명은 생략.

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

CONFIGS['last_date_time'] = data['date_time'].max()
CONFIGS['n_buildings'] = len(data['num'].unique())
```

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

### Dataset

#### Input Columns

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
    'is_holiday', 'target',
]
target_cols = ['target']
to_inverse_cols = ['mean_to_inverse', 'std_to_inverse']
input_cols = list(set(
    building_num_cols + building_info_cols + target_time_info_cols +
    time_series_cols + target_cols + to_inverse_cols
))

CONFIGS['building_num_cols'] = building_num_cols
CONFIGS['building_info_cols'] = building_info_cols
CONFIGS['target_time_info_cols'] = target_time_info_cols
CONFIGS['time_series_cols'] = time_series_cols
CONFIGS['target_cols'] = target_cols
CONFIGS['to_inverse_cols'] = to_inverse_cols
CONFIGS['input_cols'] = input_cols
```

#### Make tf.data.Dataset

여기까지는 [Multi-Task Learning 시계열 모델](https://fidabspd.github.io/2022/01/28/time_series_4-1.html)과 비교해서 사소한 수정사항들 외에는 차이가 없다고 봐도 무방하다.  
`tf.data.Dataset`을 만드는 시점부터 이전 시리즈들과 차이가 생기기 시작한다.

앞선 시계열 시리즈들에서는 `Dataset`을 구성할 때 data를 먼저 건물 단위로 나누고, 그 안에서 `flat_map`, `map` 등을 이용하여 건물별로 window를 구성하였다.  
하지만 그렇게 데이터를 구성할 경우 데이터를 `shuffle` 해주지 않으면 데이터의 순서는 0번 건물이 먼저 다 나온 뒤 1번...59번 까지 반복된다. 이렇게 데이터를 구성할 경우 rolling prediction(recursive 하게 input활용)을 수행하기에 불편함이 많다. 

그래서 현재는 데이터를 건물별 sort된 순서가 아닌, date_time별로 sort된 순서로 데이터를 이용하도록 하자. 이렇게 하면 첫날의 60개 건물에 대한 데이터가 나오고 그 다음날의 60개 건물... 순서로 데이터가 나오게 된다.  
이렇게 데이터를 구성하는 key는 `window`의 `stride`이다.

`stride` 에 대해 알아보기 위해 간단한 데이터를 만들어 테스트해보자.

```python
ds = Dataset.from_tensor_slices(np.arange(20))
ds = ds.window(size=2, shift=1, stride=2, drop_remainder=True)
ds = ds.flat_map(lambda x: x).batch(2)
list(ds.as_numpy_iterator())
```

```
[array([0, 2]),
 array([1, 3]),
 array([2, 4]),
 array([3, 5]),
 array([4, 6]),
 array([5, 7]),
 array([6, 8]),
 array([7, 9]),
 array([ 8, 10]),
 array([ 9, 11]),
 array([10, 12]),
 array([11, 13]),
 array([12, 14]),
 array([13, 15]),
 array([14, 16]),
 array([15, 17]),
 array([16, 18]),
 array([17, 19])]
```

보다시피 `stride`는 window를 구성할 때 데이터를 있는 그대로의 순서가 아닌 일부를 `stride`만큼 건너뛴 순서의 것을 다음 데이터로 보고 window를 구성한다.  
이를 이용하여 multi-task learning의 dataset을 쉽게 구성할 수 있으며, 데이터의 순서를 task의 순서가 아닌 seqence의 기준이 되는 시간의 순서로 정렬할 수 있다.

실제로 사용하기 전에 예시를 하나만 더 보자.  
task의 개수는 3개, 각 task가 가지는 seqence_length는 3인 데이터를 생성해보자. (window_size는 4를 이용하겠다.)

```python
SIZE = 4; N_TASK = 3; SEQ_LEN = 7
data = [
    list(range(N_TASK))*SEQ_LEN,
    [i//N_TASK for i in range(N_TASK*SEQ_LEN)],
]
data = np.array(data).T
data
```

```
array([[0, 0],
       [1, 0],
       [2, 0],
       [0, 1],
       [1, 1],
       [2, 1],
       [0, 2],
       [1, 2],
       [2, 2],
       [0, 3],
       [1, 3],
       [2, 3],
       [0, 4],
       [1, 4],
       [2, 4],
       [0, 5],
       [1, 5],
       [2, 5],
       [0, 6],
       [1, 6],
       [2, 6]])
```

```python
ds = Dataset.from_tensor_slices(data)
ds = ds.window(size=SIZE, shift=1, stride=N_TASK, drop_remainder=True)
ds = ds.flat_map(lambda x: x).batch(SIZE)
list(ds.as_numpy_iterator())
```

```
[array([[0, 0],
        [0, 1],
        [0, 2],
        [0, 3]]),
 array([[1, 0],
        [1, 1],
        [1, 2],
        [1, 3]]),
 array([[2, 0],
        [2, 1],
        [2, 2],
        [2, 3]]),
 array([[0, 1],
        [0, 2],
        [0, 3],
        [0, 4]]),
 array([[1, 1],
        [1, 2],
        [1, 3],
        [1, 4]]),
 array([[2, 1],
        [2, 2],
        [2, 3],
        [2, 4]]),
 array([[0, 2],
        [0, 3],
        [0, 4],
        [0, 5]]),
 array([[1, 2],
        [1, 3],
        [1, 4],
        [1, 5]]),
 array([[2, 2],
        [2, 3],
        [2, 4],
        [2, 5]]),
 array([[0, 3],
        [0, 4],
        [0, 5],
        [0, 6]]),
 array([[1, 3],
        [1, 4],
        [1, 5],
        [1, 6]]),
 array([[2, 3],
        [2, 4],
        [2, 5],
        [2, 6]])]
```

이렇게 하나의 task가 하나의 window씩을 만들도록 데이터를 구성할 수 있다.

이를 실제 데이터에 적용해보자.

```python
def mk_time_series(data, CONFIGS, is_input=False, is_time_series=False):
    if is_input:
        data = data[:-CONFIGS['target_length']*CONFIGS['n_buildings']]
    else:
        data = data[CONFIGS['window_size']*CONFIGS['n_buildings']:]
    ds = Dataset.from_tensor_slices(data)
    if is_time_series:
        if is_input:
            size = CONFIGS['window_size']
        else:
            size = CONFIGS['target_length']
        ds = ds.window(
            size=size, shift=CONFIGS['shift'],
            stride=CONFIGS['n_buildings'], drop_remainder=True
        )
        ds = ds.flat_map(lambda x: x).batch(size)
    return ds


def mk_dataset(data, CONFIGS, batch_size=None, shuffle=False):
    
    if not batch_size:
        batch_size = CONFIGS['batch_size']
    
    data = data.sort_values(['date_time', 'num'])

    building_num = data[CONFIGS['building_num_cols']]
    building_info = data[CONFIGS['building_info_cols']]
    target_time_info = data[CONFIGS['target_time_info_cols']]
    time_series = data[CONFIGS['time_series_cols']]
    to_inverse = data[CONFIGS['to_inverse_cols']]
    target = data[CONFIGS['target_cols']]

    building_num_ds = mk_time_series(building_num, CONFIGS)
    building_info_ds = mk_time_series(building_info, CONFIGS)
    target_time_info_ds = mk_time_series(target_time_info, CONFIGS)
    time_series_ds = mk_time_series(time_series, CONFIGS, is_input=True, is_time_series=True)
    to_inverse_ds = mk_time_series(to_inverse, CONFIGS)
    target_ds = mk_time_series(target, CONFIGS, is_time_series=True)

    ds = Dataset.zip((
        (
            building_num_ds,
            building_info_ds,
            target_time_info_ds,
            time_series_ds,
            to_inverse_ds
        ),
        target_ds
    ))
    if shuffle:
        ds = ds.shuffle(CONFIGS['buffer_size'])
    ds = ds.batch(batch_size).prefetch(2)
    
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
valid_ds = mk_dataset(valid, CONFIGS, batch_size=CONFIGS['n_buildings'])
test_ds = mk_dataset(test, CONFIGS, batch_size=CONFIGS['n_buildings'])
```

`mk_time_series` 부분만 제외하면 이전과 거의 흡사하다.  
다만 눈에 띄는 점 한가지는 `valid`와 `test`의 `batch_size`가 `CONFIGS`의 `batch_size`와 다르다는 것이다.  

##### batch_size

`valid`와 `test`의 `batch_size`는 `CONFIGS['n_buildings']`로 주었는데 이는 rolling prediction을 쉽게 하기 위함이다.  
`batch_size`를 `CONFIGS['n_buildings']`로 설정하면 첫번째 batch의 output을 두번째 batch의 input에 그대로 다시 사용할 수 있다.  
굳이 이렇게 사용하지 않더라도 다양한 방법으로 해결할 수 있지만 이 방법이 가장 직관적인 방법 중 하나라고 생각한다.

### Modeling

#### Customize Loss & Metric

```python
class CustomMSE(Loss):
    
    def __init__(self, target_max, name="custom_mse"):
        super(CustomMSE, self).__init__(name=name)
        self.target_max = target_max

    def call(self, y_true, y_pred):
        y_true = tf.squeeze(y_true)
        mean = tf.reshape(y_pred[:, -2], (-1, 1))
        std = tf.reshape(y_pred[:, -1], (-1, 1))
        y_pred = y_pred[:, :-2]

        y_true_inversed = y_true*std+mean
        y_pred_inversed = y_pred*std+mean
        
        y_true_inversed_scaled = y_true_inversed/self.target_max
        y_pred_inversed_scaled = y_pred_inversed/self.target_max

        mse = tf.reduce_mean((y_true_inversed_scaled-y_pred_inversed_scaled)**2)
        return mse

    
class InversedRMSE(Metric):
    
    def __init__(self, CONFIGS, name="inversed_rmse", **kwargs):
        super(InversedRMSE, self).__init__(name=name, **kwargs)
        self.inversed_mse = self.add_weight(name='inversed_mse', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.CONFIGS = CONFIGS

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, (-1, CONFIGS['target_length']))
        mean = tf.reshape(y_pred[:, -2], (-1, 1))
        std = tf.reshape(y_pred[:, -1], (-1, 1))
        y_pred = y_pred[:, :-2]

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
    dropout_1 = Dropout(0.3, name='dropout_1')(dense_1)
    outputs = Dense(CONFIGS['target_length'], name='outputs')(dropout_1)
    
    # to_inverse
    to_inverse_inputs = Input(batch_shape=(None, len(CONFIGS['to_inverse_cols'])), name='to_inverse_inputs')
    concat_to_inverse = Concatenate(name='concat_to_inverse')([outputs, to_inverse_inputs])
    
    if not model_name:
        model_name = CONFIGS['model_name']
    
    model = Model(
        inputs = [
            building_num_inputs,
            building_info_inputs,
            target_time_info_inputs,
            time_series_inputs,
            to_inverse_inputs
        ],
        outputs = concat_to_inverse,
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
Model: "recursive_prediction"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 building_num_inputs (InputLaye  [(None, 1)]         0           []                               
 r)                                                                                               
                                                                                                  
 building_info_inputs (InputLay  [(None, 7)]         0           []                               
 er)                                                                                              
                                                                                                  
 target_time_info_inputs (Input  [(None, 11)]        0           []                               
 Layer)                                                                                           
                                                                                                  
 time_series_inputs (InputLayer  [(None, 168, 12)]   0           []                               
 )                                                                                                
                                                                                                  
 building_num_layer (BuildingNu  (None, 10)          600         ['building_num_inputs[0][0]']    
 m)                                                                                               
                                                                                                  
 building_info_layer (BuildingI  (None, 32)          672         ['building_info_inputs[0][0]']   
 nfo)                                                                                             
                                                                                                  
 target_time_info_layer (Target  (None, 32)          736         ['target_time_info_inputs[0][0]']
 TimeInfo)                                                                                        
                                                                                                  
 time_series_layer (TimeSeries)  (None, 320)         2160        ['time_series_inputs[0][0]']     
                                                                                                  
 concat (Concatenate)           (None, 394)          0           ['building_num_layer[0][0]',     
                                                                  'building_info_layer[0][0]',    
                                                                  'target_time_info_layer[0][0]', 
                                                                  'time_series_layer[0][0]']      
                                                                                                  
 dense_0 (Dense)                (None, 64)           25280       ['concat[0][0]']                 
                                                                                                  
 dropout_0 (Dropout)            (None, 64)           0           ['dense_0[0][0]']                
                                                                                                  
 dense_1 (Dense)                (None, 32)           2080        ['dropout_0[0][0]']              
                                                                                                  
 dropout_1 (Dropout)            (None, 32)           0           ['dense_1[0][0]']                
                                                                                                  
 outputs (Dense)                (None, 1)            33          ['dropout_1[0][0]']              
                                                                                                  
 to_inverse_inputs (InputLayer)  [(None, 2)]         0           []                               
                                                                                                  
 concat_to_inverse (Concatenate  (None, 3)           0           ['outputs[0][0]',                
 )                                                                'to_inverse_inputs[0][0]']      
                                                                                                  
==================================================================================================
Total params: 31,561
Trainable params: 31,561
Non-trainable params: 0
__________________________________________________________________________________________________
```

모델 구성은 [Multi-Task Learning 시계열 모델](https://fidabspd.github.io/2022/01/30/time_series_4-2.html)과 완벽히 일치한다.

#### Recursive Input을 활용한 Evaluation

rolling prediction을 수행한 결과를 평가하기 위한 함수를 정의한다.  

```python
def recursive_eval(model, ds, data_usage, CONFIGS):
    
    if data_usage == 'valid':
        seq_len = str_to_dt(CONFIGS['test_start_date_time']) - \
            str_to_dt(CONFIGS['valid_start_date_time'])
    elif data_usage == 'test':
        seq_len = str_to_dt(CONFIGS['last_date_time'])+datetime.timedelta(hours=1) - \
            str_to_dt(CONFIGS['test_start_date_time'])
    seq_len = seq_len.total_seconds()/3600
    assert seq_len % CONFIGS['target_length'] == 0, \
        f'seq_len must be multiple of target_length. Now seq_len: {seq_len}, target_length: {CONFIGS["target_length"]}'

    (_, _, _, fisrt_ts, _), _ = iter(ds).next()
    ts_target = fisrt_ts[..., -1:]

    inversed_mse = 0
    for i, ((bn, bi, tti, ts, ti), y_true) in enumerate(ds):
        assert len(y_true) == CONFIGS['n_buildings'], \
            f'batch_size is {len(y_true)} now. Set batch_size same as CONFIGS["n_buildings"]'
        if i%CONFIGS['target_length'] != 0:
            continue
        ts_wo_target = ts[..., :-1]
        ts_concat = tf.concat([ts_wo_target, ts_target], axis=-1)

        y_true = tf.reshape(y_true, (CONFIGS['n_buildings'], CONFIGS['target_length']))
        y_pred = model.predict((bn, bi, tti, ts_concat, ti))
        y_pred, mean, std = y_pred[..., :-2], y_pred[..., [-2]], y_pred[..., [-1]]

        ts_target = tf.concat([
            ts_target[:, CONFIGS['target_length']:, :],
            y_pred.reshape(CONFIGS['n_buildings'], CONFIGS['target_length'], 1)
        ], axis=1)

        y_true_inversed = y_true*std+mean
        y_pred_inversed = y_pred*std+mean

        inversed_mse += tf.reduce_sum((y_true_inversed-y_pred_inversed)**2)/(seq_len*CONFIGS['n_buildings'])

    inversed_rmse = inversed_mse**0.5

    return inversed_rmse
```

[batch_size](#batch_size)에서 설명했듯이 `batch_size`를 task의 개수와 같게 설정하여, 첫 batch를 이용한 prediction 결과를 두번째 batch의 input에 포함시키고 이를 반복 수행하도록 한다. 하지만 현재 `target_length=3`, `shift=1` 이기 때문에 약간의 수정이 필요하다. 다양한 방법이 있겠지만 가장 간단하게 3번의 batch마다 이전의 prediction 값을 다음번의 input에 포함시키는 방법을 사용하도록 하자.  
이전과 마찬가지로 inversed된 rmse를 계산한다.

#### Customize Callback

모든 시계열 시리즈들에서 공통적으로 **best model**을 선정할 떄 validset을 이용했다. 그러기 위해 valid_loss, valid_metric을 모든 epoch이 끝날 때마다 계산하는 과정이 반드시 따라왔다. 그리고 이는 `model.fit()`에 `validation_data`를 넣어줌으로써 가능했다.  
하지만 현재는 바로 앞에 정의한 `recursive_eval`을 이용하여 `validation_data`를 평가해야한다. 방법은 다양하겠지만 `Callback`을 이용한 customize를 해보자.

`recursive_eval`을 이용하는 것 외에도 전 시계열 시리즈 포스팅들에서 꾸준히 사용하던 `EarlyStopping`과 `ModelCheckpoint`를 직접 구현해보자.

```python
class BestByRecursiveRMSE(Callback):

    def __init__(self, valid_ds, CONFIGS):
        super(BestByRecursiveRMSE, self).__init__()
        self.valid_ds = valid_ds
        self.CONFIGS = CONFIGS
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.best_epoch = None
        self.wait = 0
        self.stopped_epoch = 0
        self.best_train_loss = np.inf
        self.best_train_rmse = np.inf
        self.best_valid_rmse = np.inf

    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get('loss')
        train_rmse = logs.get('inversed_rmse')
        valid_rmse = recursive_eval(self.model, self.valid_ds, 'valid', self.CONFIGS)
        print(f'Epoch: {epoch}')
        print(f'\ttrain loss: {train_loss:.07f}\ttrain inversed rmse: {train_rmse:.07f}')
        print(f'\trecursive inversed valid rmse: {valid_rmse:.07f}\n')
        
        if np.less(valid_rmse, self.best_valid_rmse):
            self.best_epoch = epoch
            self.best_train_loss = train_loss
            self.best_train_rmse = train_rmse
            self.best_valid_rmse = valid_rmse
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= CONFIGS['es_patience']:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            self.model.save_weights(f'{CONFIGS["model_path"]}{CONFIGS["model_name"]}.h5')
            print(f'\nBest epoch by recursive valid rmse: {self.best_epoch}')
            print(f'\ttrain loss: {self.best_train_loss:.07f}\ttrain inversed rmse: {self.best_train_rmse:.07f}')
            print(f'\trecursive inversed valid rmse: {self.best_valid_rmse:.07f}')
```

함수명이 직관적인 편으로 코드 읽어보면 어떤 기능을 수행 하는지 파악할 수 있다.  
`Callback`을 상속받으면 정확히 어떤 기능들을 수행할 수 있는지는 워낙 광범위하기 때문에 기회가 되면 따로 다뤄보도록 하고, 지금은 사용한 것들에 대해서만 알아보자.

`on_train_begin`은 말 그대로 train이 시작될 때 호출 된다.  
여기서 각종 best score들을 기록하기 위한 변수들을 init한다.

`on_epoch_end`는 epoch이 끝날 때마다 호출된다.  
`train_loss`, `train_inversed_rmse` 두가지는 `logs`에서 불러온다.  
`valid_rmse`는 미리 정의한 `recursive_eval`를 이용하여 계산한다.  
`valid_rmse` 기준 best값이 갱신되면 `wait`을 0으로 초기화하고 해당 epoch의 loss와 metric값들을 새로운 best로 저장한다.  
`es_patience`만큼 best의 갱신이 일어나지 않으면 학습을 종료한다. 

`on_train_end`는 학습이 종료되면 호출된다.  
best model을 저장한다.

#### Train

```python
def train_model(model, train_ds, valid_ds, CONFIGS):
    
    tensorboard_callback = TensorBoard(
        log_dir = CONFIGS['tensorboard_log_path']
    )
    best_by_recursive_rmse = BestByRecursiveRMSE(valid_ds, CONFIGS)
    
    history = model.fit(
        train_ds,
        batch_size = CONFIGS['batch_size'],
        epochs = CONFIGS['epochs'],
        callbacks = [
            best_by_recursive_rmse,
            tensorboard_callback,
        ],
        verbose=0
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

recursive_valid_rmse = recursive_eval(best_model, valid_ds, 'valid', CONFIGS)
recursive_test_rmse = recursive_eval(best_model, test_ds, 'test', CONFIGS)

print(f'train_loss: {train_loss:.07f}\ttrain_rmse: {train_rmse:.07f}')
print(f'valid_loss: {valid_loss:.07f}\tvalid_rmse: {valid_rmse:.07f}')
print(f'test_loss: {test_loss:.07f}\ttest_rmse: {test_rmse:.07f}')

print(f'\nrecursive_valid_rmse: {recursive_valid_rmse:.6f}\nrecursive_test_rmse: {recursive_test_rmse:.6f}')
```

```
train_loss: 0.0001319	train_rmse: 194.2317047
valid_loss: 0.0005534	valid_rmse: 397.8041687
test_loss: 0.0003501	test_rmse: 316.4278564

recursive_valid_rmse: 448.891640
recursive_test_rmse: 387.803739
```

이전의 예측의 prediction값을 다음 예측의 input으로 recursive하게 활용하여 모델을 완성하고 평가하였다.  

이상으로 약 한달 반가량 이어온 시계열 시리즈 포스팅을 끝마친다.

## 목차

1. 시계열 데이터의 기본적인 특징과 간단한 모델
1. tf.data.Dataset을 이용한 시계열 데이터 구성
1. Multi-Input 시계열 모델
1. Multi-Task Learning 시계열 모델 (1)
1. Multi-Task Learning 시계열 모델 (2)
1. 시계열 target의 결측
1. **이전의 예측값을 다음의 input으로 recursive하게 이용**

## 원본 코드 ➞ [<span style="color:#AC1538">CODE (GitHub)</span>](https://github.com/fidabspd/time_series/blob/master/codes/6_recursive_prediction.ipynb)
