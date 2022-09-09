---
layout: single
title: "파이썬 프로그래밍으로 배우는 컴퓨터 비전 - Chapter 1"
categories: "Computer_Vision"
author: hanwool
toc: true
toc_sticky: true
---

# 파이썬 프로그래밍으로 배우는 컴퓨터 비전 - Chapter 1

이번 시간에는 컴퓨터 비전을 파이썬 프로그래밍으로 배우는 시간을 가지려고 합니다.  이 포스트의 주요 내용은  [Programming Computer Vision with Python](http://programmingcomputervision.com/) 책을 번역 및 참고하였습니다. 이 책은 총 10장 으로 이루어져있으며, 컴퓨터 비전의 주요 이론 및 알고리즘을 공부하기 좋은 책입니다. 

그럼 들어가기에 앞서, 기본적으로 이 책은 파이썬으로 프로그래밍을 하기 때문에 Python 2.6+ 버전 이상이 설치 되어 있어야 하며, 추가적으로 각 장마다 새로운 파이썬 모듈을 사용한다는 사실을 알고 계시길 바랍니다. 핵심 모듈은 [NumPy](http://numpy.scipy.org/) 와 [Matplotlib](http://matplotlib.sourceforge.net/) 이며, 추가적으로 [SciPy](http://scipy.org/) 을 사용하기도 합니다.

그럼 지금부터 1 장을 같이 배우도록 하겠습니다.

## Chapter 1. Basic Image Handling and Processing

1장은 기본적으로 이미지를 어떻게 처리하는지에 대해서 소개합니다. 이 장에서는 간단하게 이미지를 처리하기 위해서 어떤 파이썬 패키지를 사용하는지 대해서 소개합니다. 그 뿐 아니라, 이미지를 읽고, 전환 및 수정, 그리고 출력 하는 방법등에 대해서도 소개합니다. 

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


앞으로 많은 예시들에서 이미지 리스트를 처리해야 하는 경우가 있습니다.  폴더 안에 있는 이미지들의 파일 리스트를 어떻게 저장할 수 있을지 다음 예시를 보면 알 수 있습니다. imtools.py 라는 파일은 새롭게 만들어 아래 함수를 추가해주세요.

```python
import os

def get_imlist(path):
    """ Returns a list of filenames for
    all jpg images in a directory. """
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
```

### 썸네일 생성

PIL은 썸네일 생성도 가능하게 합니다. thumbnail() 함수를 사용하여 썸네일을 생성할 수 있는데, tuple을 활용하여 썸네일 사이즈를 조절할 수 있습니다.

```python
pil_im.thumbnail((128,128))
```

### 이미지 영역 복사 및 붙여넣기

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

### 크기 조절 및 회전

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
<font size="3"> 예제 1-1 PIL을 이용한 이미지 처리 예시들 </font>
</div>


예제 1-1을 보면 지금까지 한 예시들의 결과를 확인할 수 있습니다. 맨 왼쪽에 있는 사진이 원본이며, 중간에 있는 사진은 그레이 스케일 사진 마지막은 썸네일 사진 및 회전된 이미지 영역을 보여줍니다.