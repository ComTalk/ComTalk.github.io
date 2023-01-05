---
layout: single
title: "파이썬 프로그래밍으로 배우는 컴퓨터 비전 - Chapter 1 (2)"
categories: "Computer_Vision"
author: hanwool
toc: true
toc_sticky: true
---

이번 시간에는 저번시간에 이어서 1장의 컴퓨터비전의 중요한 파이썬 라이브러리인 NumPy에 대해서 알아보도록 하겠습니다. 


### 1.3 NumPy

[NumPy](https://numpy.org/doc/) 는 파이썬을 이용하여 과학적 계산을 하기에 아주 유용한 패키지 입니다. NumPy는 유용한 많은 개념들을 가지고 있는데, 예를 들어 Array 객체(벡터, 매트릭스, 이미지 등을 표현하는데 사용) 와 선형대 수학 함수들을 포함하고 있습니다. NumPy array 객체는 이 책의 대부분의 예제에 사용됩니다. Array 객체는 중요한 작업들, 예를 들어 이미지 조정, 뒤틀림, 모델 변형, 이미지 분류, 그룹화 등에 사용되는 매트릭스 곱셈, 전치, 방정식 계산, 벡터 곱셈, 그리고 정규화 등을 활용할 수 있게 해줍니다.

#### Array 이미지 묘사

우리가 저번 예시에서 이미지를 불러올 때, 이미지를 **array()** 함수를 통해 NumPy Array 객체로 변환 시켰습니다. NumPy의 Array는 다차원이며, 벡터, 매트릭스 그리고 이미지로 표현가능 합니다. Array는 파이썬의 List 와 비슷하다고 볼 수 있는데, 하지만 모든 Array의 요소는 같은 데이터 타입이여야 합니다. 우리가 구체적으로 데이터 타입을 지정하지 않아도, 데이터의 타입에 맞게 Array는 자동적으로 데이터 타입이 정해집니다.

다음 예시를 같이 보도록 하겠습니다.

```python
im = array(Image.open('data/empire.jpg'))
print(im.shape, im.dtype)

im = array(Image.open('data/empire.jpg').convert('L'),'f')
print(im.shape, im.dtype)

```

결과 값은 아래와 같이 나올겁니다.

```
(800, 569, 3) uint8
(800, 569) float32
```

첫번째 줄의 tuple 은 이미지 Array의 형태 (행, 열, 색상 채널) 를 보여주며, 다음 문자열 값은 Array 요소의 데이터 타입을 나타냅니다. 이미지는 주로 unsigned 8-bit integers(uint8) 로 부호화되며, 따라서 이미지를 Array로 변환하여도 데이터 타입은 "unit8"로 유지됨을 알 수 있습니다. 두번째 사례는 이미지를 그레이스케일로 변경 후 추가된 argument 'f' 와 함께 Array를 생성합니다. 'f' 는 floating point 타입을 줄인 명령어 입니다. 그레이스케일은 tuple에 두 값만 표현하는데 그 이유는 컬러 정보가 존재하지 않기 때문입니다. 

Array에 있는 요소들도 색인을 통해 접근 가능합니다. 좌표 i, j 그리고 색상 채널 k 값을 통해 다음과 같이 접근 가능합니다:

``` python
value = im[i,j,k]
```

다수의 요소들도 Array 슬라이싱을 통해 한번에 접근 가능합니다. 슬라이싱은 Array의 특정 구간을 반환합니다. 그레이스케일 이미지의 예시를 통해 더 자세히 알아보도록 하겠습니다.

```python
im[i,:] = im[j,:] # 행 i번의 값을 행 j번의 값으로 지정함
im[:,i] = 100 # 열 i 번까지 100 으로 값으로 지정함
im[:100,:50].sum() # 첫 행 100번까지와 열 50 번까지의 값의 총합
im[50:100,50:100] # 행 50-100, 열 50-100 (100번째는 제외)
im[i].mean() # 행 i 의 평균
im[:,-1] # 마지막 열
im[-2,:] (or im[-2]) # 마지막에서 두번째 행 
```

만약에 하나의 색인 값만 사용하면, 행의 색인으로 해석합니다.  그리고 음수 색인 값은 뒤쪽부터 계산합니다.  슬라이싱으로 픽셀 값을 접근하는 것을 잘 이해하는 것이 앞으로의 내용들을 이해하는데 필요합니다. 

#### 그레이 스케일 변형

이미지를 Array로 읽게 되면, 우리가 원하는 어떠한 수학적 작업들도 가능해집니다.  이미지를 그레이스케일로 변형시키는 간단한 예시가 있습니다.  어떤 함수 f는 0 .. 255 (혹은 0 ... 1) 의 값을 결과 값에 입력 값과 동일한 범위를 매핑 시킵니다. 다음 예시를 같이 보도록 하겠습니다.

```python
im = array(Image.open('data/empire.jpg').convert('L'))
im2 = 255 - im # 이미지 도치
im3 = (100.0/255) * im + 100 # 100 - 200 사이 값으로 고정
im4 = 255.0 * (im/255.0)**2 # 거듭 제곱
```

첫번째 사례는 그레이스케일 이미지를 도치시켰습니다. 두번째 사례는 이미지의 강도 값을 100 - 200 사이로 고정시켰습니다. 세번째 사례는 이차 함수를 적용하여, 어두운 픽셀의 값들을 더 낮췄습니다. 그림 1-1은 함수들의 결과 값을 보여주며, 그림 1-2는 결과 이미지들을 보여줍니다. 최소값과 최대값도 아래의 예시를 통해서 구할 수 있습니다:

```python
print(int(im.min()), int(im.max()))
```

최소값과 최대값의 각각의 결과 값은 아래와 같습니다.
```
2 255
0 253
100 200
0 255
```

<br/>
<div align="center">
<img src="https://github.com/ComTalk/ComTalk.github.io/raw/master/assets/images/Computer_Vision/third/python_example_1_1.png" width="700">
<br/>
<font size="3"> 그림 1-1. 그레이스케일 변형 예시. 원본은 점선으로 표현하였습니다. </font>
</div>

<br/>

<br/>
<div align="center">
<img src="https://github.com/ComTalk/ComTalk.github.io/raw/master/assets/images/Computer_Vision/third/python_example_1_2.png" width="700">
<br/>
<font size="3"> 그림 1-2. 그레이스케일 변형 결과 이미지. 첫번째 그림은 이미지 도치. 두번째 그림은 이미지를 특정 값 100-200 사이로 고정. 세번째 그림은 이차 변형 결과 이미지.</font>
</div>

<br/>

**array()** 함수의 반대의 함수는 PIL 함수의 **fromarray()** 함수입니다:

```python
pil_im = Image.fromarray(im)
```

만약에 데이터 타입을 "unit8"에서 다른 타입으로 변경하였다면, PIL 이미지로 변경하기전에 반드시 데이터 타입을 재변경하여야 합니다:

```python
pil_im = Image.fromarray(uint8(im))
```

만약에 정확히 기억이 안난다면, 안전한 방법으로 "uint8"로 변경시키는 것을 추천합니다. NumPy 는 항상 array 타입을 데이터를 표현할 수 있는 가장 최소 타입으로 변경합니다. floating point의 곱셈과 나눗셈 계산에서는 integer 타입의 Array를 float 로 변경합니다.

#### 이미지 크기 조절

NumPy Array 는 이미지와 데이터를 처리하는 데 중점적인 역할을 합니다. Array의 크기를 조절하는데 쉬운 방법이 없습니다. 따라서, PIL image 객체를 활용해서 간단하게 크기를 조절합니다. 아래 예시를 imtools.py 에 추가해주세요:

```python
def imresize(im,sz):
	""" PIL을 이용하여 Array 사이즈 조절 """
	pil_im = Image.fromarray(uint8(im))
	return array(pil_im.resize(sz))
```

#### 히스토그램 균일화

그레이스케일 변형의 아주 유용한 예시는 히스토그램 균일화 입니다. 이 변형은 그레이스케일 히스토그램을 평평하게 만들어 모든 강도 값이 최대한 균일하게 만들어 줍니다. 이러한 방식은 이미지 값을 처리하기 전 정규화 시킬때 좋습니다. 그뿐 아니라, 이미지 대비를 증가 시키는데에도 유용합니다.

이번 예시의 변형 함수는 픽셀의 누적분포함수(구간 픽셀 값을 특정 구간으로 정규화 매핑)입니다. 이번 예시도 imtools.py 에 추가해주세요:

```python
def histeq(im,nbr_bins=256):
	""" 그레이스케일 이미지의 히스토그램 균일화 """
	# 이미지 히스토그램 구하기
	imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
	cdf = imhist.cumsum() # 누적분포함수
	cdf = 255 * cdf / cdf[-1] # 정규화
	# 새 픽셀 값을 구하기 위한 선형 보간법
	im2 = interp(im.flatten(),bins[:-1],cdf)
	return im2.reshape(im.shape), cdf
```

이 함수는 그레이스케일 이미지를 받아서 BIN의 갯수를 구합니다. BIN은 히스토그램의 입력 값으로 반환 값은 픽셀 값을 매핑한 누적분포함수 와 균일화된 히스토그램을 포함한 이미지입니다.  알아둬야 할 점은 누적분포함수의 마지막 요소(index - 1)를 사용하여 0 -1 사이의 값으로 정규화 시켰다는 점입니다. 그럼 위의 함수를 이미지에 적용해 보겠습니다:

```python
from PIL import Image
from numpy import *
import imtools

im = array(Image.open('data/AquaTermi_lowcontrast.jpg').convert('L'))
im2,cdf = imtools.histeq(im)
```

그림 1-3과 그림 1-4를 통해 히스토그램 균일화의 결과 값을 확인 할 수 있습니다. 첫번째 줄은 그레이스케일의 히스토그램의 균일화 적용전과 후를 누적분포함수 매핑과 함께 확인 할 수 있습니다. 보시는 바와 같이 대비값이 증가하여 어두운 부분을 더욱 더 상세하게 확인 할 수 있습니다.

<br/>
<div align="center">
<img src="https://github.com/ComTalk/ComTalk.github.io/raw/master/assets/images/Computer_Vision/third/python_example_1_3.png" width="700">
<br/>
<font size="3"> 그림 1-3. 히스토그램 균일화 예시. 왼쪽은 원본이미지와 히스토그램. 중앙은 그레이스케일 변형 함수 도표. 오른쪽 그림은 히스토그램 균일화 후 이미지와 히스토그램</font>
</div>

<br/>

<br/>
<div align="center">
<img src="https://github.com/ComTalk/ComTalk.github.io/raw/master/assets/images/Computer_Vision/third/python_example_1_4.png" width="700">
<br/>
<font size="3"> 그림 1-4. 히스토그램 균일화 예시. 왼쪽은 원본이미지와 히스토그램. 중앙은 그레이스케일 변형 함수 도표. 오른쪽 그림은 히스토그램 균일화 후 이미지와 히스토그램</font>
</div>

<br/>

#### 이미지 주성분 분석

이미지 주성분 분석(PCA) 는 차원의 수를 감소 및 최적화하여 학습 데이터가 최대한 적은 차원에서 다양하게 표현할 수 있게 하는 유용한 기술입니다. 100 x 100 의 작은 그레이스케일 이미지도 10,000 차원을 가지고 있으며, 10,000 의 차원 공간의 점으로 인식됩니다. 메가 픽셀 이미지는 수백만의 차원을 가지고 있습니다. 그러므로, 차원 축소가 다양한 컴퓨터 비전 어플리케이션에서 사용되는게 그리 놀랍지 않을 것입니다. PCA의 결과인 투영 매트릭스는 좌표가 중요도의 내림차순인 좌표계로의 좌표 변경으로 볼 수 있습니다.

이미지 데이터에 PCA를 적용하기 위해서는 이미지를 1차원 벡터로 전환하여야 하면, NumPy의 **flatten()** 함수를 사용할 수 있수 있습니다. 

평평화된 이미지는 각 이미지에 대해 한 행씩 쌓아서 단일 매트릭스로 수집됩니다. 그런 다음 행은 지배적인 방향을 계산하기 전에 평균 이미지에 대해 상대적으로 중앙에 배치됩니다. 주성분을 구하려면 보통 특이값 분해(SVD)를 사용하지만, 차원이 높을 경우 SVD 계산이 매우 느리기 때문에 대신 사용할 수 있는 유용한 방법이 있습니다:

```python
from PIL import Image
from numpy import *

def pca(X):

	""" 이미지 주성분 분석(PCA)
	입력: X, 학습 데이터가 행의 평평한 배열로 저장된 매트릭스
	반환값: 투영 매트릭스(중요한 차원를 먼저 사용), 분산 및 평균.
	"""

	# 차원값 얻기
	num_data,dim = X.shape

	# 데이터 평균값
	mean_X = X.mean(axis=0)
	X = X - mean_X

	if dim>num_data:
		# PCA - 간결한 방법 사용
		M = dot(X,X.T) # 공분산 매트릭스
		e,EV = linalg.eigh(M) # 고유값 및 고유 벡터
		tmp = dot(X.T,EV).T # 이것이 핵심 방법
		V = tmp[::-1] # 마지막 고유 벡터가 우리가 원하는 것이므로 마지막 색인 값
		S = sqrt(e)[::-1] # 고유값이 오름차순 이므로 마지막 색인 값
		for i in range(V.shape[1]):
			V[:,i] /= S

	else:
		# PCA - SVD 사용
		U,S,V = linalg.svd(X)
		V = V[:num_data] # 처음 num_data를 반환하는 것만 의미있음

	# 투영 매트릭스, 분산 및 평균을 반환
	return V,S,mean_X
```

이 함수는 먼저 각 차원의 평균을 빼서 데이터의 중심을 잡습니다. 그런 다음 간결한 방법 또는 SVD를 사용하여 공분산 매트릭스의 가장 큰 고유 값에 해당하는 고유 벡터를 계산합니다. 그런 다음 우리는 Integer 값 n을 취하고 Integer 값 0 에서 (n – 1)의 List를 반환하는 **range()** 함수를 사용했습니다. 대체 함수로 Array를 제공하는 **arange()** 함수나 제너레이터(향상된 속도를 제공)를 제공하는 **xrange()** 함수를 사용하셔도 됩니다. 이 책에서는 **range()** 함수를 주로 사용할 것입니다.

데이터 포인트 수가 벡터의 차원보다 작을 경우, (작은) 공분산 매트릭스  <img src="https://latex.codecogs.com/svg.image?XX^{T}" title="https://latex.codecogs.com/svg.image?XX^{T}" /> 의 고유 벡터를 계산하는 방식으로 SVD 대신에 좀 전에 설명했던 간결한 방법을 사용할 것입니다.  또한, k 개의 가장 큰 고유 값(k는 원하는 차원의 수)에 해당하는 고유 벡터만 계산하여 훨씬 더 빠르게 만드는 방법도 있습니다. 이러한 방법들은 이 책의 범위를 넘어서기 때문에, 따로 다루지는 않겠습니다. 매트릭스 V의 행은 직교하고, 학습 데이터의 내림차순의 좌표 방향을 포함합니다.

글꼴 이미지를 이용해 이 방법을 사용해 보겠습니다. 파일 "fontimages.zip"은다양한 글꼴로 인쇄된 다음 스캔된 문자 "a"의  작은 썸네일 이미지가 포함되어 있습니다. 2,359개의 글꼴은 무료로 제공된 글꼴 모음으로부터 나온 것입니다. 이러한 이미지의 파일 이름들이 List(imlist) 에 저장되어 있다고 가정하고, 이전에 사용했던 코드(pca.py)를 통해 주요 구성 요소들을 다음과 같이 계산하고 표현할 수 있습니다:

```python
from PIL import Image
from numpy import *
from pylab import *
import pca
import glob
import os

# 이미지 파일이름 List 에 저장
imlist = glob.glob(os.path.join('data/fontimages/a_thumbs', '*.jpg'))

im = array(Image.open(imlist[0])) # 이미지 열기
m,n = im.shape[0:2] # 이미지 사이즈 얻기
imnbr = len(imlist) # 이미지 갯수 확인

# 매트릭스는 모든 평평한 이미지들을 생성할 수 있습니다. 
immatrix = array([array(Image.open(im)).flatten() for im in imlist],'f')

# PCA 실행
V,S,immean = pca.pca(immatrix)

# 일부 이미지 표시(평균 및 7가지 초기 모드)
figure()
gray()
subplot(2,4,1)
imshow(immean.reshape(m,n))

for i in range(7):
	subplot(2,4,i+2)
	imshow(V[i].reshape(m,n))

show()
```

<br/>

<br/>
<div align="center">
<img src="https://github.com/ComTalk/ComTalk.github.io/raw/master/assets/images/Computer_Vision/third/python_example_1_5.png" width="700">
<br/>
<font size="3"> 그림 1-5. 평균 이미지(왼쪽 위) 와 처음 7가지 모드; 즉 편동이 가장 많은 방향들을 표현</font>
</div>

<br/>


여기서 fontimages 는 data 폴더에 저장되어 있다고 가정하겠습니다.  먼저 **reshape()** 함수를 사용하여 이미지를 1차원에서 원본 차원으로 변환하여야 합니다. 예시 코드를 실행하면, 그림 1-5와 같이 하나의 창에 8개의 이미지가 표시 됩니다. PyLab의 **subplot()** 함수를 통해 하나의 창에 여러 개의 이미지를 배치했습니다.

#### Pickle 모듈 사용

일부 결과나 데이터를 나중에 사용하기 위해 저장하고 싶다면, 파이썬에서는 아주 유용한 Pickle 모듈을 제공합니다. Pickle은 거의 모든 Python 객체를 가져와서 문자열 표현으로 변환할 수 있습니다. 이러한 과정을 우리는 pickling 이라고 부릅니다. 문자열 표현으로 부터 객체를 재구성하는 것을 반대로 unpickling 이라고 부릅니다. 이러한 문자열 표현은 쉽게 저장되거나 전송 될 수 있습니다.

하나의 예를 통해 이러한 방식을 설명하도록 하겠습니다. 바로전 섹션에서 사용했던 이미지 평균과 글꼴 이미지의 주요 구성 요소들을 저장한다고 가정해 보겠습니다:

```python
import pickle

# 평균 및 주요 구성 요소들 저장
f = open('font_pca_modes.pkl', 'wb')
pickle.dump(immean,f)
pickle.dump(V,f)
f.close()
```

보시다시피 여러 객체들을 한 파일에 저장할 수 있습니다. ".pkl" 파일에 사용할 수 있는 프로토콜은 여러가지가 있으며, 확실하지 않은 경우에는 바이너리 파일로 읽고 쓰는 것이 가장 좋습니다. 다른 Python 섹션에서 데이터를 불러오려면, **load()** 함수를 사용하면 됩니다:

```python
import pickle

# 평균 및 주요 구성 요소들 불러오기
f = open('font_pca_modes.pkl', 'rb')
immean = pickle.load(f)
V = pickle.load(f)
f.close()
```

중요한 점은 객체의 순서가 저장할 때랑 동일해야 한다는 점입니다! 표준 Pickle 모듈과 완벽히 호환되는 C로 작성된 cpickle 의 최적화된 버전도 존재합니다. 자세한 내용은 Pickle 모듈 설명서 페이지 에서 [http://docs.python.org/library/pickle.html](http://docs.python.org/library/pickle.html) 확인할 수 있습니다.

이 책에서는 파일 읽기 및 쓰기를 처리하기 위해 **with** 문을 사용할 것입니다. 이것은 Python 2.5 버전에서 소개된 구조로, 파일 열기 및 닫기를 자동으로 처리해 줍니다.(파일이 열려 있는 동안 오류가 발생하더라도 처리해줍니다.) 위에 소개된 저장 및 불러오기를 **with()** 문을 이용하면 다음과 같습니다:

```python
import pickle

# 파일 열고 저장
with open('font_pca_modes.pkl', 'wb') as f:
	pickle.dump(immean,f)
	pickle.dump(V,f)

# 파일 열고 불러오기
with open('font_pca_modes.pkl', 'rb') as f:
	immean = pickle.load(f)
	V = pickle.load(f)
```

처음 보신 분들은 헷갈릴 수도 있지만, 아주 유용한 구조입니다. 이러한 구조가 마음에 들지 않으시다면, 그냥 **open** 이나 **close** 문을 사용하시면 됩니다. 또한, NumPy 에는 Pickle을 사용하는 대신에, 데이터에 복잡한 구조가 포함되지 않는 경우(예: 이미지에서 클릭한 포인트 List) 에는 텍스트 파일을 읽고 쓸 수 있는 간단한 함수들도 있습니다. Array x를 저장하기 위해서:

```python
savetxt('test.txt',x,'%i')
```

마지막 매개 변수는 Integer 형식을 사용해야 함을 나타냅니다. 마찬가지로, 읽기는 다음과 같이 수행됩니다:

```python
x = loadtxt('test.txt')
```

좀 더 자세히 알고 싶으시다면 온라인 설명서를 참고하세요. [http://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html](http://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html)

추가로 NumPy에는 Array를 저장하고 불러오는 전용 함수들이 있습니다. 좀 더 자세히 알고 싶으시다면, 온라인 설명서에 있는 **save()** 함수나 **load()** 를 읽어 보시길 바랍니다.


이번 시간에는 파이썬 라이브러리 NumPy에 대해서 간단히 소개 하는 시간을 가졌습니다. 다음 포스팅에서는 SciPy에 대해서 알아보도록 하겠습니다.

출처 : Solem, J.E. (no date) Programming Computer Vision with python, Programming Computer Vision with Python. Available at: http://programmingcomputervision.com/ (Accessed: October 7, 2022). 