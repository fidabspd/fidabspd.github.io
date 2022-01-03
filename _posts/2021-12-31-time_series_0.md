---
layout: post
title: 시계열 예측
tags: [Time-Series, Tensorflow, Keras]
excerpt_separator: <!--more-->
---

시계열 데이터를 다루고 TensorFlow로 모델을 구성하는 데 있어 겪는 문제들과 그에 대한 가이드를 시리즈 형식으로 정리해보려 하며 그 시리즈의 첫번째입니다.  
아주 간단한 시계열 모델에 점차 살을 붙여가는 형식으로 진행될 예정이며 목차는 다음과 같습니다.
<!--more-->

## 목차

0. **시계열 데이터의 기본적인 특징과 간단한 모델**
1. tf.data.Dataset 이용
2. 시계열 target 외에 다른 데이터를 함께 이용
3. 독립적인 여러개의 시계열 데이터에 대한 예측
4. 시계열 데이터 scaling
5. 시계열 target의 결측
6. 이전의 예측값을 다음의 input으로 recursive하게 이용

***

## 시작하기 전에

- **본 포스팅 시리즈의 목적은 시계열 예측의 정확성을 높이기 위함이 아닙니다.**
- **단지 시계열 데이터를 다루고 예측함에 있어 부딪히는 여러 문제들과 그를 어떻게 해결하면 좋을 지 가이드를 제시하기 위함입니다.**
- **데이터는 [DACON 전력사용량 예측 AI 경진대회](https://dacon.io/competitions/official/235736/overview/description)의 데이터를 사용합니다.** 감사합니다 DACON

***

## 시계열 데이터의 특징

#### 대부분의 정형 데이터는 ...

머신러닝 공부를 시작하는 단계에서 접하는 대부분의 정형 데이터들은 하나의 row가 하나의 데이터를 나타내는 경우가 많습니다. (편의상 앞으로 이런 데이터를 OneRowOneData를 축약하여 **OROD**이라고 부르도록 하겠습니다. ~~제가 방금 멋대로 정한 단어입니다. 다른데서 썼다가는 조금 민망할 수도 있습니다.~~) 튜토리얼로 많이 사용되는 캐글의 타이타닉 생존자 예측 데이터를 보면 하나의 row가 한명의 승객정보를 담고 있고, 이를 통해 해당 승객이 생존했는지 생존하지 못했는지 예측합니다. 하나의 row에 예측을 위해 사용될 input과 예측 target 값이 모여있습니다.  

#### 하지만 시계열 데이터는 ...

시계열 데이터는 하나의 row가 하나의 데이터를 나타내지 않습니다. 시간의 흐름에 따라 row 방향으로 데이터를 쌓는다면 이전 row들을 이용해 다음 row들을 예측합니다.  
만약 본인이 위에 언급한 하나의 row가 하나의 데이터를 나타내는 데이터만 다룰 수 있는 상황이라 한들 시계열 데이터를 다룰 수 없는 것은 아닙니다. 다루기 익숙한 모양(하나의 row가 하나의 데이터)에 맞춰 데이터를 재구성 하면 됩니다.

#### 그런데 하나의 데이터는 뭘까

앞서 말한 OROD데이터는 input과 target이 정해져있습니다. (물론 파생변수를 만들거나 target에 변형을 가하면 달라지긴 합니다.) 하지만 시계열 데이터는 input과 target에 대한 자유도가 상당히 높습니다. 이를 확실하게 구성하기 위해서는 다음과 같은 내용을 미리 정해야합니다.

- 이전 몇 time 을 사용하여 다음 값을 예측할 것인지
- 다음 몇 time 동안의 값을 예측할 것인지
- 예측을 몇 time 단위로 띄어가며 진행할 것인지 (데이터의 양 조절)

그리고 각각을 앞으로 다음과 같이 칭하겠습니다. (완전히 약속된 단어는 아니지만 통상 이렇게 지칭하면 의사소통에 문제는 없습니다.)

- window_size
- target_length
- interval

이들은 일종의 hyper parameter로 볼 수 있으며 상황과 데이터에 맞게 조절해야합니다.

#### window_size, target_length, interval 예시

비트코인의 가격이 분단위로 기록된 10일치 데이터를 가지고 있다고 가정하겠습니다.   
이때 ```window_size = 120, target_length = 10, interval = 5```를 다른말로 하면  
```이전 2시간의 데이터를 이용하여 이후 10분을 예측하며 5분마다 예측한다.```
라고 할 수 있습니다.

#### 총 row 수

위의 예시를 이용하여 다음 시계열 데이터를 OROD데이터로 구성한다면 몇개의 row가 생기게 될까요?  
이는 다음과 같이 계산 할 수 있습니다.

```
총 seqence길이(10일치): 10*24*60 = 14400
window_size + target_length: 120 + 10 = 130
총 row 수: {(14400 - 130) / 5} + 1 = 2855
```

일반화 해보자면

```
총 row 수: ((seqence_length - (window_size+target_lenght)) / interval) + 1
```

이렇게 표현이 가능합니다.  
혹시라도 ```((seqence_length - (window_size+target_lenght))```가 ```interval```로 나눠 떨어지지 않는 것이 걱정이라면 각 숫자를 조정하거나, 앞부분 데이터를 조금 잘라내거나, 딱 한곳의 interval만 임의로 조정함으로서 해결할 수 있습니다.

## 데이터에 적용

앞으로 사용하는 모든 데이터는 [DACON 전력사용량 예측 AI 경진대회](https://dacon.io/competitions/official/235736/overview/description)의 전력 사용량 데이터를 가공하여 사용합니다.  
가공된 데이터는 다음과 같습니다.  

```python
data
```

![data](/assets/img/posts/2021-12-31-time_series_0/data.png)  
20년 6월 1일부터 20년 8월 24일까지 시간단위로 데이터가 있습니다.

#### train, valid, test 구성

간단하게 testset은 마지막날인 8월 24일로 하고 이를 예측하는 모델을 구성해보도록 하겠습니다.  
그런데 validset은 어떻게 구성하면 좋을까요?  
구성에 고려해야할 것은 여러가지가 있지만 크게 두가지정도만 짚겠습니다.

1. validset이 시계열 데이터의 주기를 모두 포함하였는지
2. 전체 데이터 중 어떤 데이터를 valid로 사용할 지

우선 **1번**고려사항에 관해서는, validset은 당연히 시계열 데이터의 주기를 모두 포함하는 것이 좋습니다.  
어떤 도로의 교통량을 예측하는 데 validset을 주말 시간대로만 구성한다면 주말만 잘 맞추는 모델이 되겠죠?  
하지만 현실적으로 모든 주기를 모두 포함하기는 쉽지 않기 때문에 상황에 따른 적절한 분배가 필요합니다.

**2번**고려사항에 관해서는, 데이터를 구성하는 방법은 크게 두가지 정도로 생각할 수 있으며 장단점은 다음과 같습니다.

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

    2번 방법을 사용하여 valid score에 대한 신뢰도를 높이면서 이를 train에도 활용하는 방법도 있긴 합니다.  
    trainset만으로 모든 hyperparameter에 대한 tuning을 마치고 validset을 trainset에 포함시켜 다시 train하는 방법입니다.  
    다만 이는 trainset만으로 hyperparameter를 tuning하였기 때문에 validset이 train에 포함되는 순간 결과가 이상해지는 경우가 생길 수 있습니다.  
    이런 이유로 딥러닝 모델에는 특히 추천하고 싶지 않습니다.

전체적인 흐름보다 주기에 훨씬 의존적인 모습을 보인다면 **1번**을 사용할 수도 있겠습니다. 다만 대부분의 경우에 **2번**방법을 추천드립니다.

저는 **2번**방법을 사용하여 20년 8월 23일은 valid_data, 20년 8월 24일은 test_data, 나머지는 train_data로 사용하여 data를 OROD형태로 재구성 해보겠습니다.  
(weekly와 monthly 주기는 일단은 고려하지 않겠습니다.)

- ```window_size```: 7*24 (7일)
- ```target_length```: 3 (3시간)
- ```interval```: 1 (1시간)

```python
CONFIGS = {
    'test_lenght': 24,
    'valid_start_index': 1992,
    'test_start_index': 2016,
    
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

재구성 된 데이터는 다음과 같습니다.

```python
train
```

![train](/assets/img/posts/2021-12-31-time_series_0/train.png)  

```python
seqence_length = 2040 - 48  # valid와 test를 빼줬기 때문에 -48
window_size = 24*7
target_lenght = 3
interval = 1

nrow = ((seqence_length - (window_size+target_lenght)) / interval) + 1
print(f'총 row 수: {nrow}')
# 총 row 수: 1822
```

계산대로 총 1822개의 row가 생긴 모습입니다.

## Modeling

아주 간단한 NN model을 만들고 훈련해 보겠습니다.  

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
        loss = 'mse',
        optimizer = optimizer,
    )
    
    if print_summary:
        model.summary()
    
    return model


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
CONFIGS['model_name'] = 'super_basic'
CONFIGS['batch_size'] = 64
CONFIGS['learning_rate'] = 1e-4
CONFIGS['epochs'] = 100
CONFIGS['es_patience'] = 10

model = set_model(CONFIGS, print_summary=True)
history = train_model(model, train, valid, CONFIGS)
```

![model_summary](/assets/img/posts/2021-12-31-time_series_0/model_summary.png)  

best model을 load하여 test를 예측하고 성능을 비교해보겠습니다.

```python
best_model = set_model(CONFIGS, model_name='best_'+CONFIGS['model_name'])
best_model.load_weights(f'{CONFIGS["model_path"]}{CONFIGS["model_name"]}.h5')

X_train, y_train = train[CONFIGS['input_cols']], train[CONFIGS['target_cols']]
X_valid, y_valid = valid[CONFIGS['input_cols']], valid[CONFIGS['target_cols']]
X_test, y_test = test[CONFIGS['input_cols']], test[CONFIGS['target_cols']]

y_train_pred = best_model.predict(X_train)
y_valid_pred = best_model.predict(X_valid)
y_test_pred = best_model.predict(X_test)

mse = MeanSquaredError()

train_loss = mse(y_train, y_train_pred).numpy()
valid_loss = mse(y_valid, y_valid_pred).numpy()
test_loss = mse(y_test, y_test_pred).numpy()

print(f'train_loss: {train_loss}')
print(f'valid_loss: {valid_loss}')
print(f'test_loss: {test_loss}')

# train_loss: 7653.4296875
# valid_loss: 10269.60546875
# test_loss: 14098.6875
```

데이터 구성부터 간단한 모델링까지 마쳤습니다.
**(모델 성능을 높이기 위한 고민을 담은 포스팅은 아니기 때문에 당장 모델 성능은 신경쓰지 않습니다.)**

처음부터 정독하셨다면 이런저런 의문점들이 생기는 부분이 생기셨을 수 있습니다. 예상되는 것들을 적어보자면

- 매번 데이터 구성을 저렇게 매뉴얼하게 할 것인가?
- scaling은 안해도 되나?
- testset의 시간은 24시간인데 target_length를 3으로 구성하면 알지 못한다고 가정한 testset의 target이 input으로 들어간거 아닌가?

등등 외의 여러개가 있을 것으로 예상됩니다.  

모든 의문점을 해결할 수는 없겠지만 이런 의문들에 대한 내용은 앞서 적혀있던 목차에 따라 이어질 포스팅에서 다룰 예정입니다.
