---
title: Time Series - tf.data.Dataset을 이용한 시계열 데이터 구성
tags: Time-Series, Tensorflow, Keras
---

[시계열 데이터의 기본적인 특징과 간단한 모델](https://fidabspd.github.io/2022/01/04/time_series_1.html)에서 기본적인 시계열 데이터의 특징을 알아보고 간단한 모델을 만들었다.  
이에 이어 시리즈의 두번째 내용으로 tf.data.Dataset의 전체적인 설명과 이를 이용한 데이터 구성을 해보자.

## 목차

1. [시계열 데이터의 기본적인 특징과 간단한 모델](https://fidabspd.github.io/2022/01/04/time_series_1.html)
1. [**tf.data.Dataset을 이용한 시계열 데이터 구성**](https://fidabspd.github.io/2022/01/10/time_series_2.html)
1. [Multi-Input 시계열 모델](https://fidabspd.github.io/2022/01/18/time_series_3.html)
1. [Multi-Task Learning 시계열 모델 (1)](https://fidabspd.github.io/2022/01/28/time_series_4-1.html)
1. [Multi-Task Learning 시계열 모델 (2)](https://fidabspd.github.io/2022/01/30/time_series_4-2.html)
1. [시계열 target의 결측](https://fidabspd.github.io/2022/02/10/time_series_5.html)
1. [Recursive Input Prediction](https://fidabspd.github.io/2022/02/17/time_series_6.html)

## 원본 코드 ➞ [<span style="color:#AC1538">CODE (GitHub)</span>](https://github.com/fidabspd/time_series/tree/master/codes/2_tf_dataset.ipynb)

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
    - **이번 포스팅에서는 `num`이 1인 건물의 `target` 만을 사용한다.**

***

## Manual한 데이터 구성

시계열 데이터 첫번째 시리즈 [시계열 데이터의 기본적인 특징과 간단한 모델](https://fidabspd.github.io/2022/01/04/time_series_1.html)에서 sequential하게 주어진 시계열 데이터를 하나의 row가 하나의 데이터를 나타내도록 형태를 변형하는 작업을 거쳤다.  

![orod](/post_images/time_series_2/orod.png)  

딱 봐도 같은 데이터가 여러번 반복되고있고 메모리 사용이 비효율 적일 것이라는게 눈에 보인다. 그리고 이를 만드는 과정 자체도 수고롭다.  
tf.data를 이용해 이 모든 것을 해결할 수 있다. 메모리를 효율적으로 사용하고 만드는 과정도 간편하게 만들 수 있다.

## tf.data

TensorFlow 공식 document를 읽어보면 tf.data에 대한 설명은 다음과 같다.

> The tf.data API enables you to build complex input pipelines from simple, reusable pieces. For example, the pipeline for an image model might aggregate data from files in a distributed file system, apply random perturbations to each image, and merge randomly selected images into a batch for training. The pipeline for a text model might involve extracting symbols from raw text data, converting them to embedding identifiers with a lookup table, and batching together sequences of different lengths. The tf.data API makes it possible to handle large amounts of data, read from different data formats, and perform complex transformations.

간단하게 요약하자면 입력 파이프라인 빌드를 쉽게, 빠르게 할 수 있다는 내용이다.  
한번에 모든 데이터를 구성하여 메모리에 올려놔야하는 feed_dict 방식에 비해 tf.data는 dataset 자체에서 batch와 shuffle, cache, prefetch 등을 지원하기 때문에 훨씬 빠르고 효율적이다.

### How to use

코드를 보고 기능을 하나씩 짚어보자.  
아래 데이터를 사용한다. (지난번 포스팅의 데이터와 같다.)  

![data](/post_images/time_series_2/data.png)  

```python
def mk_dataset(data, shuffle=False):
    
    X = data[:-CONFIGS['target_length']]
    y = data[CONFIGS['window_size']:]
    
    X_ds = Dataset.from_tensor_slices(X)
    X_ds = X_ds.window(CONFIGS['window_size'], shift=1, drop_remainder=True)
    X_ds = X_ds.flat_map(lambda x: x).batch(CONFIGS['window_size'])
    
    y_ds = Dataset.from_tensor_slices(y)
    y_ds = y_ds.window(CONFIGS['target_length'], shift=1, drop_remainder=True)
    y_ds = y_ds.flat_map(lambda x: x).batch(CONFIGS['target_length'])
    
    ds = Dataset.zip((X_ds, y_ds))
    if shuffle:
        ds = ds.shuffle(512)
    ds = ds.batch(CONFIGS['batch_size']).cache().prefetch(2)
    
    return ds

CONFIGS['valid_start_index'] = 1992
train = data.loc[:CONFIGS['valid_start_index'], 'target']
train_ds = mk_dataset(train, shuffle=True)
```

### Functions

사용한 기능들은 다음과 같다.

- `from_tensor_slices`
- `window`
- `flat_map`
- `batch`
- `zip`
- `shuffle`
- `cache`
- `prefetch`

#### from_tensor_slices

공식 document의 설명은 다음과 같다.

> Creates a Dataset whose elements are slices of the given tensors.  
> 
> The given tensors are sliced along their first dimension. This operation preserves the structure of the input tensors, removing the first dimension of each tensor and using it as the dataset dimension. All input tensors must have the same size in their first dimensions.

가장 바깥쪽 dim을 이용해 데이터를 잘라 dataset의 dim으로 활용한다.

**Example**

```python
a = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
a_ds = Dataset.from_tensor_slices(a)

print('a:', a, sep='\n')
print('')
print('a_ds:', a_ds, sep='\n')
print('')
for i in a_ds:
    print(i)
print('')
print(list(a_ds.as_numpy_iterator()))
```

**Result**

```
a:
tf.Tensor(
[[1 2 3]
 [4 5 6]
 [7 8 9]], shape=(3, 3), dtype=int32)

a_ds:
<TensorSliceDataset shapes: (3,), types: tf.int32>

tf.Tensor([1 2 3], shape=(3,), dtype=int32)
tf.Tensor([4 5 6], shape=(3,), dtype=int32)
tf.Tensor([7 8 9], shape=(3,), dtype=int32)

[array([1, 2, 3], dtype=int32), array([4, 5, 6], dtype=int32), array([7, 8, 9], dtype=int32)]
```

처음 `tf.constant`로 만든 데이터인 `a`는 출력하면 데이터의 세부 내용이 전부 출력되지만 이를 `from_tensor_slices`를 이용해 변환한 `a_ds`는 data shape, type만 출력될 뿐 세부 내용이 출력되지 않는다.  
이의 세부 내용을 보기 위해서는 위와 같이 for문을 이용해 출력하거나 `as_numpy_iterator`등 을 이용하여 출력한다. (이 외에 다른 방법들도 있다.)

모든 tf.data는 이런 특징을 가진다.

#### window

`window`의 기능은 다음 예시로 쉽게 이해가 된다.

**Example**

```python
a = list(range(10))
a_ds = Dataset.from_tensor_slices(a)
a_window = a_ds.window(size=5, shift=1, drop_remainder=True)

print('a:', a, sep='\n')
print('')
print('after window:')
for w in a_window:
    print(list(w.as_numpy_iterator()))
```

**Result**

```
a:
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

after window:
[0, 1, 2, 3, 4]
[1, 2, 3, 4, 5]
[2, 3, 4, 5, 6]
[3, 4, 5, 6, 7]
[4, 5, 6, 7, 8]
[5, 6, 7, 8, 9]
```

`window`는 이렇게 시계열 데이터를 구성할 때 유용하다.  
`size`와 `shift`는 시계열 첫번째 시리즈 포스팅에서 말한 **window_size**와 **shift**를 의미한다.  
`drop_reaminder`는 다음 예시로 쉽게 이해 가능하다. `drop_remainder=False`로 주게 되면 결과는 다음과 같다.

```python
a_window = a_ds.window(size=5, shift=1, drop_remainder=False)

print('after window:')
for w in a_window:
    print(list(w.as_numpy_iterator()))
```

```
after window:
[0, 1, 2, 3, 4]
[1, 2, 3, 4, 5]
[2, 3, 4, 5, 6]
[3, 4, 5, 6, 7]
[4, 5, 6, 7, 8]
[5, 6, 7, 8, 9]
[6, 7, 8, 9]
[7, 8, 9]
[8, 9]
[9]
```

그런데 `a_window`를 출력함에 있어 이상한 점이 하나 보인다. for문을 사용했음에도 그 안에 `as_numpy_iterator`를 또 사용한다.  
for문 만을 사용하면 출력 결과는 다음과 같다.

```python
print('after window:')
for w in a_window:
    print(w)
```

```
after window:
<_VariantDataset shapes: (), types: tf.int32>
<_VariantDataset shapes: (), types: tf.int32>
<_VariantDataset shapes: (), types: tf.int32>
<_VariantDataset shapes: (), types: tf.int32>
<_VariantDataset shapes: (), types: tf.int32>
<_VariantDataset shapes: (), types: tf.int32>
```

이를 보면 알 수 있듯이 window로 분할된 sub_dataset들도 tf.data의 특성을 가진다.  
다른 방법으로도 출력해보자.

```python
a_window = a_ds.window(size=5, shift=1, drop_remainder=True)

print('after window:')
for w in a_window:
    for i in w:
        print(i)
    print('')
```

```
after window:
tf.Tensor(0, shape=(), dtype=int32)
tf.Tensor(1, shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)
tf.Tensor(3, shape=(), dtype=int32)
tf.Tensor(4, shape=(), dtype=int32)

tf.Tensor(1, shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)
tf.Tensor(3, shape=(), dtype=int32)
tf.Tensor(4, shape=(), dtype=int32)
tf.Tensor(5, shape=(), dtype=int32)

tf.Tensor(2, shape=(), dtype=int32)
tf.Tensor(3, shape=(), dtype=int32)
tf.Tensor(4, shape=(), dtype=int32)
tf.Tensor(5, shape=(), dtype=int32)
tf.Tensor(6, shape=(), dtype=int32)

tf.Tensor(3, shape=(), dtype=int32)
tf.Tensor(4, shape=(), dtype=int32)
tf.Tensor(5, shape=(), dtype=int32)
tf.Tensor(6, shape=(), dtype=int32)
tf.Tensor(7, shape=(), dtype=int32)

tf.Tensor(4, shape=(), dtype=int32)
tf.Tensor(5, shape=(), dtype=int32)
tf.Tensor(6, shape=(), dtype=int32)
tf.Tensor(7, shape=(), dtype=int32)
tf.Tensor(8, shape=(), dtype=int32)

tf.Tensor(5, shape=(), dtype=int32)
tf.Tensor(6, shape=(), dtype=int32)
tf.Tensor(7, shape=(), dtype=int32)
tf.Tensor(8, shape=(), dtype=int32)
tf.Tensor(9, shape=(), dtype=int32)
```

예상과는 조금 다르다.  
처음 공부할 때 제 예상 출력 결과는 다음과 같았다.

```
tf.Tensor([0 1 2 3 4], shape=(5,), dtype=int32)
tf.Tensor([1 2 3 4 5], shape=(5,), dtype=int32)
tf.Tensor([2 3 4 5 6], shape=(5,), dtype=int32)
tf.Tensor([3 4 5 6 7], shape=(5,), dtype=int32)
tf.Tensor([4 5 6 7 8], shape=(5,), dtype=int32)
tf.Tensor([5 6 7 8 9], shape=(5,), dtype=int32)
```

이렇게 예상과 다른 출력 결과가 나오는 이유는 `a_ds`가 이렇게 생겼기 때문이다.

```python
for i in a_ds:
    print(i)
```

```
tf.Tensor(0, shape=(), dtype=int32)
tf.Tensor(1, shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)
tf.Tensor(3, shape=(), dtype=int32)
tf.Tensor(4, shape=(), dtype=int32)
tf.Tensor(5, shape=(), dtype=int32)
tf.Tensor(6, shape=(), dtype=int32)
tf.Tensor(7, shape=(), dtype=int32)
tf.Tensor(8, shape=(), dtype=int32)
tf.Tensor(9, shape=(), dtype=int32)
```

이렇게 tf.Tensor들을 5개씩 묶은것 뿐이기 때문이다. 요컨데 이런 형태로 만든것.

```
[tf.Tensor(0, shape=(), dtype=int32)
 tf.Tensor(1, shape=(), dtype=int32)
 tf.Tensor(2, shape=(), dtype=int32)
 tf.Tensor(3, shape=(), dtype=int32)
 tf.Tensor(4, shape=(), dtype=int32)]

[tf.Tensor(1, shape=(), dtype=int32)
 tf.Tensor(2, shape=(), dtype=int32)
 tf.Tensor(3, shape=(), dtype=int32)
 tf.Tensor(4, shape=(), dtype=int32)
 tf.Tensor(5, shape=(), dtype=int32)]

...

```

즉 이를 데이터 셋으로 활용하기 위해서는 **바로 위와 같이 생긴 녀석들을 `tf.Tensor([0 1 2 3 4], shape=(5,), dtype=int32)` 이런 형태로 차원을 줄이고 flatten해주는 과정**이 필요하다.  
이를 위해 `flat_map`과 `batch`를 사용한다.

#### flat_map

기본적으로는 흔히 사용하는 파이썬 기본 기능 `map`과 같은 기능을 한다.  
하지만 그에 더해 차원을 하나 축소시킨다.

```python
a = [[1,2,3],[4,5,6],[7,8,9]]
a_ds = Dataset.from_tensor_slices(a)
a_fm = a_ds.flat_map(lambda x: Dataset.from_tensor_slices(x**2))
for i in a_fm:
    print(i)
```

```
tf.Tensor(1, shape=(), dtype=int32)
tf.Tensor(4, shape=(), dtype=int32)
tf.Tensor(9, shape=(), dtype=int32)
tf.Tensor(16, shape=(), dtype=int32)
tf.Tensor(25, shape=(), dtype=int32)
tf.Tensor(36, shape=(), dtype=int32)
tf.Tensor(49, shape=(), dtype=int32)
tf.Tensor(64, shape=(), dtype=int32)
tf.Tensor(81, shape=(), dtype=int32)
```

**map**  
map도 있다. 위 코드에서 `flat_map`을 `map`으로 바꾸면 다음과 같다.

```python
a_fm = a_ds.map(lambda x: Dataset.from_tensor_slices(x**2))
for i in a_fm:
    print(i)
```

```
<_VariantDataset shapes: (), types: tf.int32>
<_VariantDataset shapes: (), types: tf.int32>
<_VariantDataset shapes: (), types: tf.int32>
```

#### batch

`batch`는 특정 개수만큼씩의 데이터를 하나의 tf.Tensor 안에 묶어준다.  
앞선 예시에서 `batch`적용 전 `flat_map`까지 적용한 결과는 다음과 같다.

```python
a = list(range(10))
a_ds = Dataset.from_tensor_slices(a)
a_window = a_ds.window(size=5, shift=1, drop_remainder=True)
a_fm = a_window.flat_map(lambda x: x)

for i in a_fm:
    print(i)
```

```
tf.Tensor(0, shape=(), dtype=int32)
tf.Tensor(1, shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)
tf.Tensor(3, shape=(), dtype=int32)
tf.Tensor(4, shape=(), dtype=int32)
tf.Tensor(1, shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)
tf.Tensor(3, shape=(), dtype=int32)
tf.Tensor(4, shape=(), dtype=int32)
tf.Tensor(5, shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)
tf.Tensor(3, shape=(), dtype=int32)
tf.Tensor(4, shape=(), dtype=int32)
tf.Tensor(5, shape=(), dtype=int32)
tf.Tensor(6, shape=(), dtype=int32)
tf.Tensor(3, shape=(), dtype=int32)
tf.Tensor(4, shape=(), dtype=int32)
tf.Tensor(5, shape=(), dtype=int32)
tf.Tensor(6, shape=(), dtype=int32)
tf.Tensor(7, shape=(), dtype=int32)
tf.Tensor(4, shape=(), dtype=int32)
tf.Tensor(5, shape=(), dtype=int32)
tf.Tensor(6, shape=(), dtype=int32)
tf.Tensor(7, shape=(), dtype=int32)
tf.Tensor(8, shape=(), dtype=int32)
tf.Tensor(5, shape=(), dtype=int32)
tf.Tensor(6, shape=(), dtype=int32)
tf.Tensor(7, shape=(), dtype=int32)
tf.Tensor(8, shape=(), dtype=int32)
tf.Tensor(9, shape=(), dtype=int32)
```

이를 `window_size`만큼 잘라 하나의 tf.Tensor 안에 묶어주기 위해 batch를 사용한다.

```python
a_batch = a_fm.batch(5)

for b in a_batch:
    print(b)
```

```
tf.Tensor([0 1 2 3 4], shape=(5,), dtype=int32)
tf.Tensor([1 2 3 4 5], shape=(5,), dtype=int32)
tf.Tensor([2 3 4 5 6], shape=(5,), dtype=int32)
tf.Tensor([3 4 5 6 7], shape=(5,), dtype=int32)
tf.Tensor([4 5 6 7 8], shape=(5,), dtype=int32)
tf.Tensor([5 6 7 8 9], shape=(5,), dtype=int32)
```

이렇게 window를 거친 데이터를 원하는 형태로 flatten 해주기 위해 `flat_map`과 `batch`를 통한 일종의 트릭을 사용했고, 성공적으로 마쳤다.  
(개인적으로는 특정 차원을 직접 flatten할 수 있는 기능이 추가되면 좋겠다. 그러면 이렇게 복잡한 과정을 거치지 않아도 될 것 같다... 언젠가는 추가되길!)

#### zip

`zip`은 이름 그대로 짝지어 묶어주는 기능을 한다.

```python
a = [[1,2,3],[4,5,6],[7,8,9]]
b = list(range(3))
a_ds = Dataset.from_tensor_slices(a)
b_ds = Dataset.from_tensor_slices(b)
ab_zip = Dataset.zip((a_ds, b_ds))

for i in ab_zip:
    print(i)
```

```
(<tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 3], dtype=int32)>, <tf.Tensor: shape=(), dtype=int32, numpy=0>)
(<tf.Tensor: shape=(3,), dtype=int32, numpy=array([4, 5, 6], dtype=int32)>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)
(<tf.Tensor: shape=(3,), dtype=int32, numpy=array([7, 8, 9], dtype=int32)>, <tf.Tensor: shape=(), dtype=int32, numpy=2>)
```

#### shuffle

`shuffle`은 말 그대로 데이터를 섞어주는 기능을 한다.  
근데 `ds.shuffle(512)`이와 같이 parameter를 하나 받는다. 단지 섞어주는 것뿐인데 인수를 왜 받는지, 무슨 역할을 하는지 의문이 생긴다.

이 인수는 `buffer_size`.
이 인수의 기능을 알기에 앞서 tf.data를 사용하는 이유중 큰 부분은 효율적인 메모리 사용에 있다.  
대용량 데이터를 처리할 때, 극단적으로 데이터 용량이 RAM 용량보다 큰 경우라면 shuffle을 하기 위해 모든 데이터를 전부 가져와 shuffle 하는 것은 당연히 불가능하다.  
쉽게 이해하자면 `buffer_size`는 이를 위해 해당 사이즈만큼의 데이터만을 가져와서 shuffle을 해주는 것이라고 생각하면 된다.  

위와 같이 `buffer_size`를 512로 설정하면 훈련 과정에서는 다음과 같은 과정이 반복된다.

- 512개를 가져와서 섞은 뒤 batch 개수만큼 내보낸다.
- batch개수만큼 내보낸 데이터를 다시 채우고 섞는다.
- 다시 섞은 데이터를 batch 개수만큼 내보낸다.
- 반복 ...

#### cache

`cache` 역시 이름 그대로 cache형태로 데이터를 저장한다.

#### prefetch

`prefetch`도 위의 것들과 마찬가지로 데이터를 미리(pre)+가져(fetch)온다.  
훈련시에 특정 데이터를 필요로 하는 순간 그 데이터를 가져오기 시작한다면 훈련하는 과정은 그 사이에 놀게 된다. 이를 방지해주기 위해서 사용한다.  
데이터가 필요한 순간에 해당 데이터가 미리 준비되어 있도록 병렬적으로 데이터를 가져온다.

이 외에도 입력 파이프라인을 최적화하는 다양한 방법들이 있다. 기회가 된다면 이 또한 다뤄보도록 하겠다.

## 실제 데이터에 적용

우선 실제 데이터는 기능을 알아보는데 사용했던 예시들과는 다르게 `X`와 `y`가 구분되어야 한다.  
또한 flatten을 위한 트릭으로 사용했던 batch 외에 실제로 훈련을 위한 batch를 구성해야한다.  

이를 위한 과정 외에는 위에서 사용했던 기능 그대로 사용하면 된다.

```python
def mk_dataset(data, shuffle=False):
    
    X = data[:-CONFIGS['target_length']]  # 마지막 target_length만큼은 input으로 사용되지 않음
    y = data[CONFIGS['window_size']:]  # 처음 window_size만큼은 target으로 사용되지 않읍
    
    X_ds = Dataset.from_tensor_slices(X)
    X_ds = X_ds.window(CONFIGS['window_size'], shift=1, drop_remainder=True)
    X_ds = X_ds.flat_map(lambda x: x).batch(CONFIGS['window_size'])
    
    y_ds = Dataset.from_tensor_slices(y)
    y_ds = y_ds.window(CONFIGS['target_length'], shift=1, drop_remainder=True)
    y_ds = y_ds.flat_map(lambda x: x).batch(CONFIGS['target_length'])
    
    ds = Dataset.zip((X_ds, y_ds))  # X와 y를 짝 지음
    if shuffle:
        ds = ds.shuffle(512)  # train data만 shuffle 해줄 것
    ds = ds.batch(CONFIGS['batch_size']).cache().prefetch(2)  # 훈련시 사용되는 batch_size
    
    return ds
```

위 과정을 통해 tf.data.Dataset을 만들어 보았다. 이를 이용한 train 및 evaluate 과정은 전체 원본 코드에서 확인하실 수 있다.

현재까지는 target column만을 이용한 시계열 데이터 구성과 모델링을 진행했다.  
다음 포스팅에서는 target column 뿐만 아니라 다른 데이터까지 함께 사용해보도록 하자.

## 목차

1. [시계열 데이터의 기본적인 특징과 간단한 모델](https://fidabspd.github.io/2022/01/04/time_series_1.html)
1. [**tf.data.Dataset을 이용한 시계열 데이터 구성**](https://fidabspd.github.io/2022/01/10/time_series_2.html)
1. [Multi-Input 시계열 모델](https://fidabspd.github.io/2022/01/18/time_series_3.html)
1. [Multi-Task Learning 시계열 모델 (1)](https://fidabspd.github.io/2022/01/28/time_series_4-1.html)
1. [Multi-Task Learning 시계열 모델 (2)](https://fidabspd.github.io/2022/01/30/time_series_4-2.html)
1. [시계열 target의 결측](https://fidabspd.github.io/2022/02/10/time_series_5.html)
1. [Recursive Input Prediction](https://fidabspd.github.io/2022/02/17/time_series_6.html)

## 원본 코드 ➞ [<span style="color:#AC1538">CODE (GitHub)</span>](https://github.com/fidabspd/time_series/tree/master/codes/2_tf_dataset.ipynb)
