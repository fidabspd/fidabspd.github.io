---
title: 오렌지가 되어버린 침착맨의 '55도발 왜 하냐고'
tags: [Computer-Vision, OpenCV, dlib]
---

최근 흥미에 초점을 둔 공부를 시작했다. 이에 첫 프로젝트로 [어노잉 오렌지](https://namu.wiki/w/%EC%96%B4%EB%85%B8%EC%9E%89%20%EC%98%A4%EB%A0%8C%EC%A7%80) 직접 만들기를 진행하기로 했다.  

어노잉 오렌지 만들기라는 주제를 보자마자 떠올린건 *[55도발 왜 하냐고](https://www.youtube.com/watch?v=KGQEnx67lsU)의 침착맨을 오렌지로 만든다면 너무 재밌겠는데...?*  

그렇게 시작된 **오렌지가 되어버린 침착맨의 '55도발 왜 하냐고'** 만들기

## 원본 코드 ➞ [<span style="color:#AC1538">CODE (GitHub)</span>](https://github.com/fidabspd/toy/blob/master/annoying_orange/codes/main.py)

## Reference

유튜브 **빵형의 개발도상국** 채널의 **[내 얼굴로 어노잉 오렌지 만들기](https://www.youtube.com/watch?v=9VYUXchrMcM&t=204s)** 영상을 따라하되 약간의 변형과 추가만 했다.  
좋은 강의 감사합니다!

그리고 좋은 소스를 준 침착맨. 감사하다~

## CODE

### Libraries

```python
import cv2
import dlib
from imutils import face_utils, resize
import numpy as np
from moviepy.editor import VideoFileClip
```

- `cv2`: 동영상 불러오기, 저장, 이미지 합성 등 `OpenCV`라는 이름답게 다양한 CV분야에 사용
- `dlib`: 이미지에서 얼굴 탐색, 모델을 이용하여 face landmark 검출 등 수행
- `imutils`: A series of convenience functions to make basic image processing functions such as translation, rotation, resizing, skeletonization, and displaying Matplotlib images easier with OpenCV and both Python 2.7 and Python 3. [출처](https://github.com/PyImageSearch/imutils)
- `moviepy`: MoviePy is a Python module for video editing, which can be used for basic operations [출처](https://zulko.github.io/moviepy/)

### Set Configs

```python
RESOURCES_DIR = '../resources/'
RESULT_DIR = '../result/'
SIZE = (1753, 1753)
EYE_WIDTH = 342
LEFT_EYE_POS = (342, 685)
RIGHT_EYE_POS = (856, 685)
MOUTH_WIDTH = 856
MOUTH_POS = (616, 1096)
FRAME = 29.96
```

- 사용할 이미지, 영상, 모델 등을 저장한 디렉토리 주소
- 오렌지 사진의 크기에 맞춰 눈, 입을 배치할 위치
- 최종적으로 만드는 영상의 프레임

### Load

```python
orange_img = cv2.imread(RESOURCES_DIR+'orange.jpg')
orange_img = cv2.resize(orange_img, dsize=SIZE)

detector = dlib.get_frontal_face_detector()  # 사진 전체 중 얼굴자체를 detecting
predictor = dlib.shape_predictor(RESOURCES_DIR+'shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(RESOURCES_DIR+'why55_origin.mov')
```

- 오렌지 이미지
- 얼굴 랜드마크 인식 모델
- 오렌지에 합성할 55도발 영상

#### 얼굴 랜드마크 인식 모델

모델은 따로 학습하지 않고 기존에 만들어져있는 모델을 사용한다.  
[이곳](https://github.com/davisking/dlib-models)에서 다운로드 가능하다. 이 중 사용할 모델은 `shape_predictor_68_face_landmarks`이며 `dlib`을 이용해 사용한다.  

![face_landmarks](/post_images/why55_chimrangeee/face_landmarks.png)

위 사진은 모델이 검출하는 얼굴의 landmark를 표시한 것이다.  
모델을 사용하면 0~67번까지 총 68개의 landmark들을 검출하게 되며 사진 내의 픽셀 위치 값들을 리턴해준다.

### 합성 후 새로운 영상

```python
img_array = []

while cap.isOpened():
    ret, img = cap.read()  # 프레임을 한개씩 가져옴. 프레임을 제대로 읽었다면 ret값은 True

    if not ret:
        break

    # 한 이미지에 있는 모든 얼굴들의 좌표들을 가져옴
    faces = detector(img)

    result = orange_img.copy()

    if len(faces) > 0:
        face = faces[0]  # 얼굴이 하나니까 0번째만

        shape = predictor(img, face)
        shape = face_utils.shape_to_np(shape)

        # left eye
        le_x1 = shape[36, 0]
        le_y1 = shape[37, 1]
        le_x2 = shape[39, 0]
        le_y2 = shape[41, 1]
        le_margin = int((le_x2 - le_x1) * 0.35)

        # right eye
        re_x1 = shape[42, 0]
        re_y1 = shape[43, 1]
        re_x2 = shape[45, 0]
        re_y2 = shape[47, 1]
        re_margin = int((re_x2 - re_x1) * 0.35)

        # mouth
        mouth_x1 = shape[48, 0]
        mouth_y1 = shape[50, 1]
        mouth_x2 = shape[54, 0]
        mouth_y2 = shape[57, 1]
        mouth_margin = int((mouth_x2 - mouth_x1) * 0.15)

        left_eye_img = img[le_y1-le_margin:le_y2+le_margin, le_x1-le_margin:le_x2+le_margin].copy()
        right_eye_img = img[re_y1-re_margin:re_y2+re_margin, re_x1-re_margin:re_x2+re_margin].copy()
        mouth_img = img[mouth_y1-mouth_margin:mouth_y2+mouth_margin, mouth_x1-mouth_margin:mouth_x2+mouth_margin].copy()

        left_eye_img = resize(left_eye_img, width=EYE_WIDTH)
        right_eye_img = resize(right_eye_img, width=EYE_WIDTH)
        mouth_img = resize(mouth_img, width=MOUTH_WIDTH)

        # 합성
        result = cv2.seamlessClone(
            left_eye_img,
            result,
            np.full(left_eye_img.shape[:2], 255, left_eye_img.dtype),
            LEFT_EYE_POS,
            cv2.MIXED_CLONE
        )

        result = cv2.seamlessClone(
            right_eye_img,
            result,
            np.full(right_eye_img.shape[:2], 255, right_eye_img.dtype),
            RIGHT_EYE_POS,
            cv2.MIXED_CLONE
        )

        result = cv2.seamlessClone(
            mouth_img,
            result,
            np.full(mouth_img.shape[:2], 255, mouth_img.dtype),
            MOUTH_POS,
            cv2.MIXED_CLONE
        )
    
    img_array.append(result)
```

- 두 눈과 입의 위아래 양옆의 끝 점을 landmark index를 통해 잡아내고 적당한 margin을 더해 직사각형 모양으로 자연스럽게 자름
- `cv.seamlessClone`을 통해 오렌지에 하나씩 합성
- 영상의 처음부터 끝까지 모든 프레임에 대해 이를 반복하여 새로운 `img_array`를 만듦
- 영상에 얼굴이 검출되지 않을경우 수행하지 않음

### 비디오 저장

```python
out = cv2.VideoWriter(RESULT_DIR+'why55_chimrangeee_no_audio.avi', cv2.VideoWriter_fourcc(*'DIVX'), FRAME, SIZE)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()


videoclip = VideoFileClip(RESOURCES_DIR+"why55_origin.mov")
audioclip = videoclip.audio

new_videoclip = VideoFileClip(RESULT_DIR+"why55_chimrangeee_no_audio.avi")
new_videoclip.audio = audioclip

new_videoclip.write_videofile(RESULT_DIR+"why55_chimrangeee_done.mp4")
```

- 완성된 새로운 영상인 `img_array`를 저장
- 기존 영상의 audio를 새로운 영상에 그대로 붙여 저장

## 그리고 완성된 결과물..!

![chimrangeee](/post_images/why55_chimrangeee/chimrangeee.png)

# ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ

## 영상을 보고싶다면 ➞ [<span style="color:#AC1538">클릭</span>](https://youtu.be/S10aTEiQyAg)

아 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 너무 만족스럽다 ㅋㅋㅋㅋㅋㅋㅋ

침착맨의 팬으로서는 둘도 없는 웃음벨이다 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ  

## 보완점

### 합성된 눈과 입이 흔들린다.

사실 흔들리는것 때문에 B급 감성이 더해져 더 웃긴것 같아 당장은 만족스럽긴 하지만 ㅋㅋㅋㅋㅋㅋㅋ  
그래도 결과물의 퀄리티를 놓고 보면 확실히 아쉬운 부분이다.

오렌지에 위치시킬 눈과 입의 픽셀위치는 fix 되어있기 때문에 흔들리는 원인은 crop 하면서 눈 크기와 입 크기에 따라 crop 되는 이미지 자체의 크기가 달라지기 때문이라고 볼 수 있을 것 같다.  
크기뿐만 아니라 얼굴을 움직이면서 말하기 때문에 이도 영향을 미치는 것으로 보인다.

### 얼굴이 움직이는 방향대로 오렌지가 움직이지 않는다.

얼굴 움직이는대로 움직이지 않는 것 또한 B급 감성인듯 하여 만족스럽지만 위에서 말한 것 처럼 이를 해결하면 합성된 눈과 입이 흔들리는 현상도 어느정도 커버가 가능할 것으로 보인다.  
완벽하게 똑같이 움직일 수는 없겠지만 비슷하게 움직이게 하는 방법은 간단하다 생각한다.  
원본 얼굴 이미지상 두 눈과 입의 위치를 상대값으로 받아 오렌지의 상대값에 그대로 넣어주면 쉽게 해결될 듯 하다.

### 손이나 자막이 순간적으로 얼굴을 가리면 오렌지에 합성되는 얼굴 또한 순간적으로 사라진다.

두가지 관점에서 접근이 가능할 것으로 보인다. 첫번째는 손이나 자막으로 얼굴이 가려진 프레임 자체를 결측값을 바라보는 것. 두번째는 손이나 자막으로 가려진 부분을 결측값으로 바라보는 것.  
첫번째 관점으로 프레임 자체를 결측값으로 바라본다면, 손이 얼굴을 가리기 직전과 직후의 사진으로 적당한 interpolation을 통해 가장 간단하게 해결할 수 있을 것처럼 생각되지만 사실 이는 image manifold에서 벗어나기 때문에 절대 안될 것이다. 프레임 드롭된 부분의 이미지가 앞 뒤 이미지를 통해 어떻게 영상으로 이어질지 예측하는 과정이 있어야할 것으로 보이는데 상당히 복잡한 과정이 필요하지 않을까 생각된다.
두번째 관점으로 가려진 부분을 결측값으로 바라보는 것 역시 이를 어떻게 채울지는 당장은 모르겠다.

### 눈, 입을 crop하는 로직이 이미지가 뒤집히면 제대로 crop을 제대로 수행할지 의문이다.

눈과 입을 잘라내는 로직을 보면 얼굴을 똑바로 봤을 때 위, 아래, 왼쪽, 오른쪽의 가장 바깥쪽 landmark를 가져와 잘라낸다.  
만약 뒤집힌 이미지나 영상을 사용한다면 위, 아래, 왼쪽, 오른쪽이 함께 뒤집힐 것인데 이에 대한 대비가 필요할 것으로 보인다.

### face landmark 모델이 어떤 원리로 작동하는지 알지 못한다.

추가적인 공부가 필요한 부분.  
사실 오늘의 포스팅은 머신러닝에 대한 공부라기보단 `cv2`, `dlib`, `moviepy` 등을 이렇게 활용할 수 있다! 정도.  

기회가 된다면 face landmark 모델은 어떤 방식으로 학습되었는지, 원리가 무엇인지 등을 알아보도록 하자.

## 원본 코드 ➞ [<span style="color:#AC1538">CODE (GitHub)</span>](https://github.com/fidabspd/toy/blob/master/annoying_orange/codes/main.py)
