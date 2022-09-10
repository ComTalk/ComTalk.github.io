---
layout: single
title: "파이썬 프로그래밍으로 배우는 컴퓨터 비전 - Chapter 1"
categories: "Computer_Vision"
author: hanwool
toc: true
toc_sticky: true
---

이번 시간에는 컴퓨터 비전을 파이썬 프로그래밍으로 배우는 시간을 가지려고 합니다.  이 포스트의 주요 내용은  [Programming Computer Vision with Python](http://programmingcomputervision.com/) 책을 번역 및 참고하여 작성했습니다. 이 책은 총 10장 으로 이루어져있으며, 컴퓨터 비전의 주요 이론 및 알고리즘을 공부하기 좋은 책입니다. 

그럼 들어가기에 앞서, 기본적으로 이 책은 파이썬으로 프로그래밍을 하기 때문에 Python 2.6+ 버전 이상이 설치 되어 있어야 하며, 추가적으로 각 장마다 새로운 파이썬 모듈을 사용한다는 사실을 알고 계시길 바랍니다. 핵심 모듈은 [NumPy](http://numpy.scipy.org/) 와 [Matplotlib](http://matplotlib.sourceforge.net/) 이며, 추가적으로 [SciPy](http://scipy.org/) 을 사용하기도 합니다. 저는 이 포스팅에서 Python 3.7 버전을 사용하였습니다.

그럼 지금부터 1 장을 같이 배워 보도록 하겠습니다.

## Chapter 1. Basic Image Handling and Processing

1장은 기본적으로 이미지를 처리하는 다양한 방법에 대해서 소개합니다. 그리고 이 장에서는 간단하게 이미지를 처리하기 위해서 어떤 파이썬 패키지를 사용하는지 대해서 소개합니다. 그 뿐 아니라, 이미지를 읽고, 전환 및 수정, 그리고 출력 하는 방법등에 대해서도 소개합니다. 

### 1.1 PIL - The Python Imaging Library

Python Imaging Library (PIL) 은 이미지 처리 및 조작, 예를 들어 크기 조절, 자르기, 회전, 색 변경 등의 기능들을 제공합니다. PIL은 [https://pillow.readthedocs.io/en/stable/](https://pillow.readthedocs.io/en/stable/) 에서 무료로 제공됩니다.

지금부터 파이썬 예제 코드등을 통해서 PIL의 기능들을 설명하고자 합니다. 예제 코드들에 사용하는 자료들은 [http://programmingcomputervision.com/](http://programmingcomputervision.com/) 에서 다운로드 가능합니다.

먼저 PIL에 있는 Image 모듈에 대해서 설명하겠습니다. Image 모듈은 이미지들을 읽고 쓰는게 가장 많이 사용하는 모듈입니다. 

이미지를 읽기 위해서 Image 모듈의 open 함수를 이용하였습니다.

```python
from PIL import Image
pil_im = Image.open('data/empire.jpg')
```
반환 값 pil_im 은 PIL image 객체 입니다.

그리고 convert() 함수를 통해 이미지의 색상을 변경할 수 있습니다. 예를 들어서, 그레이스케일로 변경하고 싶으면, convert('L')을 추가하면 됩니다.

```python
pil_im = Image.open('data/empire.jpg').convert('L')
```

#### 이미지 포맷 변경

save() 함수를 이용하여 PIL 은 이미지를 다양한 포맷으로 변경 및 저장 가능합니다.  아래 예시는 이미지 파일 리스트를 받아서 모두 JPEG 파일로 변경하는 방법을 보여줍니다.

```python
from PIL import Image
import os

    for infile in filelist:
        outfile = os.path.splitext(infile)[0] + ".jpg"
        if infile != outfile:
            try:
                Image.open(infile).save(outfile)
            except IOError:
                print("cannot convert : ", infile)
```

코드를 간단히 설명하자면, open() 함수는 이미지 파일을 읽어 PIL image 객체로 생성하며, save() 함수는 PIL image 객체를 주어진 파일 이름으로 저장하는 기능을 합니다. 새로운 파일이름은 기존의 파일이름에 확장자만 jpg 로 변경됩니다. 예외 처리 문을 활용하여 이미 JPEG 이미지인 경우를 확인하며, 포맷 변경이 실패할 경우 에러 메세지를 출력하도록 하였습니다.

<br/>
앞으로 많은 예시들에서 이미지 리스트를 처리해야 하는 경우가 있습니다.  폴더 안에 있는 이미지들의 파일 리스트를 어떻게 저장할 수 있을지 다음 예시를 보면 알 수 있습니다. imtools.py 라는 파일은 새롭게 만들어 아래 함수를 추가해주세요.

```python
import os

def get_imlist(path):
    """ Returns a list of filenames for
    all jpg images in a directory. """
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
```

#### 썸네일 생성

PIL은 썸네일 생성도 가능하게 합니다. thumbnail() 함수를 사용하여 썸네일을 생성할 수 있는데, tuple을 활용하여 썸네일 사이즈를 조절할 수 있습니다.
```python
pil_im.thumbnail((128,128))
```

#### 이미지 영역 복사 및 붙여넣기

crop() 함수를 이용해서 이미지의 한 영역을 잘라낼 수 있습니다:

```python
box = (100,100,400,400)
region = pil_im.crop(box)
```

이미지 영역은 4-tuple (left, upper, right, lower)  로 정의됩니다.  PIL은 좌표 (0, 0) 기준이 맨 왼쪽 위 코너입니다. 추가적으로 추출한 이미지의 영역을 회전을 시킨 이후에 paste() 함수를 통해 다시 붙여넣기 할 수 있습니다.

```python
region = region.transpose(Image.ROTATE_180)
pil_im.paste(region,box)
```

#### 크기 조절 및 회전

resize() 함수에 새로운 사이즈 tuple 값을 제공함으로써 크기를 조절할 수 있습니다:

```python
out = pil_im.resize((128,128))
```
이미지 회전은 rotate() 함수를 통해서 시계방향으로 회전이 가능합니다:
```python
out = pil_im.rotate(45)
```

<br/>
<div align="center">
<img src="https://github.com/ComTalk/ComTalk.github.io/raw/master/assets/images/Computer_Vision/python_example_1_1.png" width="700">
<br/>
<font size="3"> 그림 1-1. PIL을 이용한 이미지 처리 예시들 </font>
</div>

<br/>

그림 1-1을 보면 지금까지 한 예시들의 결과를 확인할 수 있습니다. 맨 왼쪽에 있는 사진이 원본이며, 중간에 있는 사진은 그레이 스케일 사진 마지막은 썸네일 사진 및 회전된 이미지 영역을 보여줍니다.
<br/>

### 1.2 Matplotlib

Matplotlib 은 수학 공식, 그래프 도표화 및 이미지에 점, 선 및 곡선을 그리기에 PIL 보다 훨씬 적합한 그래픽 라이브버리라고 할 수 있습니다. Matplotlib은 이 책의 고화질의 그림들에 이용되었습니다. 그리고 Matplotlib 의 PyLab 인터페이스는 사용자에게 다양한 함수들을 제공하여 plots을 만들수 있게 해줍니다. Matplotlib 은 오픈소스로 [https://matplotlib.org/](https://matplotlib.org/) 에서 자세한 설명이 되어있습니다. 이 책에서는 간단한 예시를 통해서 어떤 함수들이 주로 사용되는지 보여드리겠습니다.

#### 이미지, 점 및 선 도표화

이미 우리는 막대 그래프, 파이 차트 그리고 산점도 등을 만들 수 있지만, 오직 소수의 명령어들만이 컴퓨터 비전에 필요합니다.  가장 중요한 것은 우리는 관심 지점, 관련성 그리고 점과 선만으로 감지된 객채등을 보여주고 싶어합니다. 아래 예시는 이미지에 소수의 점과 선을 도표화 시킨 예시입니다.

```python
from PIL import Image
from pylab import *

# read image to array
im = array(Image.open('data/empire.jpg'))

# plot the image
imshow(im)

# some points
x = [100,100,400,400]
y = [200,500,200,500]

# plot the points with red star-markers
plot(x,y,'r*')

# line plot connecting the first two points
plot(x[:2],y[:2])

# add title and show the plot
title('Plotting: "data/empire.jpg"')
show()
```

이 코드는 이미지를 도표화 한 다음에 붉은색의 네 개의 점을 x와 y 축 주어진 좌표 값들로 표시하고, 처음 두 개의 붉은 점을 기준으로 파란색의 선을 그리는 코드 예시 입니다. 그림 1-2의 왼쪽 그림이 위의 코드의 결과 입니다. show() 함수는 그림을 윈도우 화면에 나타내주는 기능을 합니다. show() 함수는 한 코드에 한 번만 불러야 하며, 주로 코드의 끝에서 불러줍니다. 주의할 점은 PyLab의 좌표의 원점 기준이 왼쪽 위 코너인 점이며, 이는 이미지와 동일합니다. 좌표축은 디버깅에 유용하게 사용되며, 좌표축을 지우고 싶으면 아래 코드를 추가하면 됩니다.

```python
axis('off')
```

위의 코드를 추가하면 그림1-2의 오른쪽 그림의 결과를 출력합니다. 

<br/>
<div align="center">
<img src="https://github.com/ComTalk/ComTalk.github.io/raw/master/assets/images/Computer_Vision/python_example_1_2.png" width="700">
<br/>
<font size="3"> 그림 1-2. Matplotlib 을 이용한 도표 예시들 </font>
</div>

<br/>
그리고 도표화의 색상 및 스타일의 다양한 옵션들이 있습니다. 가장 유용하게 사용되는 명령어들은 표 1-1,1-2 그리고 1-3 에서 확인할 수 있습니다. 사용 방법은 다음과 같습니다.

```python
plot(x,y) # default blue solid line
plot(x,y,'r*') # red star-markers
plot(x,y,'go-') # green line with circle-markers
plot(x,y,'ks:') # black dotted line with square-markers
```

<br/>
<div align="center">
<font size="3"> 표 1-1. PyLab의 도표화의 기본 색상 명령어들  </font><br/>
<img src="https://github.com/ComTalk/ComTalk.github.io/raw/master/assets/images/Computer_Vision/python_table_1_1.png">
<br/>
</div>

<div align="center">
<font size="3"> 표 1-2. PyLab의 도표화의 기본 스타일 명령어들 </font><br/>
<img src="https://github.com/ComTalk/ComTalk.github.io/raw/master/assets/images/Computer_Vision/python_table_1_2.png">
<br/>
</div>

<div align="center">
<font size="3"> 표 1-3. PyLab의 도표화의 기본 생산 명령어들 </font><br/>
<img src="https://github.com/ComTalk/ComTalk.github.io/raw/master/assets/images/Computer_Vision/python_table_1_3.png">
<br/>
</div>

#### 이미지 윤곽선 및 히스토그램

이번에는 이미지 윤곽선 및 히스토그램을 그려보도록 하겠습니다.  iso-contours 를 시각화 하는 것은 아주 유용할 수 있습니다. 윤곽선을 그리기 위해서는 그레이 스케일 이미지가 필요한데, 그 이유는 윤곽선은 모든 좌표 [x, y] 의 단일 값을 취하기 때문입니다. 예시 코드는 아래와 같습니다.

```python
from PIL import Image
from pylab import *

# read image to array
im = array(Image.open('data/empire.jpg').convert('L'))

# create a new figure
figure()

# don't use colors
gray()

# show contours with origin upper left corner
contour(im, origin='image')
axis('equal')
axis('off')
show()
```

먼저 그레이스케일 이미지가 필요하므로 PIL 의 convert() 함수를 이용하여 그레이스케일로 이미지를 변경하였습니다. 그 이후에 contour() 함수를 이용하여 이미지 윤곽선을 그립니다. 코드의 결과는 그림 1-3의 왼쪽 그림입니다.

이미지 히스토그램은 픽셀 값의 분포를 도표화 한 것입니다. 히스토그램의 간격은 BIN 의 크기에 따라 결정됩니다. 만약 BIN의 크기가 16이고 픽셀값의 범위가 0부터 255까지 256개일 경우,  0-256 범위는 16개의 BIN으로 구분될 것입니다. 각각의 BIN의 값은 이 한 BIN 범위에 대해 이미지가 가지고 있는 픽셀 수를 나타냅니다.  이미지 히스토그램의 시각화는 hist() 함수를 통해 가능합니다.  

```python
figure()
hist(im.flatten(),128)
show()
```

hist() 함수의 두 번째 인자는 빈의 갯수를 나타냅니다. 기억해야 할 점은 이미지를 flatten() 함수를 통해서 평평하게 만드는 것을 먼저 수행해야 합니다. 왜냐하면 hist() 함수는 1차원 배열을 입력값으로 받기 때문입니다. flatten() 함수는 어떤 배열이든 행 방향의 일차원 배열로 변경시켜주는 기능을 수행합니다. 코드의 결과값은 그림 1-3의 오른쪽 그림입니다.

<br/>
<div align="center">
<img src="https://github.com/ComTalk/ComTalk.github.io/raw/master/assets/images/Computer_Vision/python_example_1_3.png" width="700">
<br/>
<font size="3"> 그림 1-3.  Matplotlib을 이용한 이미지 윤곽선 및 히스토그램 시각화 </font>
</div>
<br/>

#### 상호적 주석달기

때때로 사용자들도 어플리케이션과 상호작용을 통해 학습 데이터에 주석을 달아야 할 필요가 있습니다. 예를 들자면, 이미지에 점들을 표시하는 것을 말합니다. PyLab 은 ginput() 과 같은 간단한 함수로 이러한 기능을 수행 가능하도록 합니다. 

```python
from PIL import Image
from pylab import *

im = array(Image.open('data/empire.jpg'))
imshow(im)

print('Please click 3 points')
x = ginput(3)

print('you clicked:', x)
show()
```

이 코드는 이미지를 도표화 한 이후에 사용자가 세 개의 점을 이미지에 표시할 때까지 기다립니다. 세 개의 점들의 좌표들은 리스트 x 에 저장됩니다. 그림 1-4는 코드의 예시를 보여줍니다. 

<br/>
<div align="center">
<img src="https://github.com/ComTalk/ComTalk.github.io/raw/master/assets/images/Computer_Vision/python_example_1_4.png" width="700">
<br/>
<font size="3"> 그림 1-4.  PyLab을 이용한 상호적 주석달기</font>
</div>
<br/>

이번 시간에는 파이썬 라이브러리 PIL 과 Matplotlib의 함수들에 대해서 간단히 소개 하는 시간을 가졌습니다. 다음 포스팅에서는 NumPy 와 SciPy 에 대해서 알아보도록 하겠습니다.