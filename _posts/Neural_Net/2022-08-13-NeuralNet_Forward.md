---
layout: single  
title: 자연어 처리를 위한 신경망 (1) - 순전파
categories: "Neural_Net"
use_math: true  
author: goeon
toc: true
tock-sticky: true
---

앞으로 자연어처리에 대한 포스팅을 연재하기에 앞서, 핵심 요소인 *신경망*에 대해 알아보겠습니다. 이번 포스팅은 [밑바닥부터 시작하는 딥러닝2](https://www.hanbit.co.kr/store/books/look.php?p_code=B8950212853)를 참조하여 작성 되었습니다. 그럼 시작해보죠.

## 신경망 소개

Neural Network(신경망)에 대해 많이 들어보셨을 겁니다. 인간의 두뇌를 모사하여 데이터를 처리하도록 만든 것인데요. 신경망은 입력층에서 출력층으로 향하는 순전파(forward)와, 반대로 출력층에서 입력층으로 향하는 역전파(backward)가 있습니다. 예를 들어 개, 고양이 사진을 보고 어떤 동물인지 예측하는 신경망이 있다고 하면, 이미지를 입력으로 받아 순전파를 통해 어떤 동물인지를 예측합니다. 그리고 실제 정답과 비교하여 잘 맞추었는지 비교해보고 앞으로 더 잘 맞출 수 있도록 역전파를 통해 신경망의 학습 파라미터를 업데이트 합니다. 이번 포스팅을 통해 신경망이 어떻게 예측하는 지 알아보도록 하죠.  

## 신경망의 예측(추론)

이번 포스팅은 신경망이 어떻게 결과를 내는지에 해당하는 신경망의 예측에 대해 알아보겠습니다. 아래의 그림은 간단한 신경망 그림입니다. 원은 노드(뉴런), 화살표는 가중치를 뜻합니다. 2차원의 입력이 들어와서 가중치와 연산을 통해 4차원이 되고, 또다시 가중치와 연산을 통해 3차원의 결과값을 출력한다는 것 정도로 이해하시면 됩니다. 

<img src="https://user-images.githubusercontent.com/13113652/184499463-9456fd08-89ba-4bfb-aa6c-9ee3e0d971f1.png" width="35%" height="35%">



입력층에서 은닉층으로 데이터가 흐르면서 실제로 어떠한 일이 발생하는 지 수식을 통해 보다 정확하게 알아보겠습니다.  

<img width="506" alt="image" src="https://user-images.githubusercontent.com/13113652/184500539-35980c63-12c0-419c-acee-c86bb40fd5d9.png">


여기서 **$x$는 입력**, **$w$는 가중치(weight)**, **$b$는 편향(bias)**, **$h$는 은닉층 노드**를 나타냅니다.  결국 그림의 화살표를 통해 일어나는 일은 **행렬연산**입니다. $h_1$만 똑 떼내어 보면 $h_1 = x_{1}w_{11}+x_{2}w_{21}+b_1$과 같은 식일 텐데요.  

물론 더 정확히 표현하면 이렇게 출력된 행렬연산을 그대로 사용하는 것이 아니라  비선형성(non-linearlity)을 추가하기 위해 활성화 함수(activation function)를 적용한 값이 전달됩니다.  식을 보다 간단하게 표현하면 $h = xW+b$로 나타냅니다. (보통 소문자는 벡터, 대문자는 행렬을 뜻합니다.)  위 식은 (1x2)x(2x4)=(1x4) 의 꼴을 갖는 행렬연산입니다.  보통 여러개의 샘플(미니배치)을 한번에 연산 하는데 N개의 샘플이 있는 경우 마찬가지로 (Nx2)x(2x4)=(Nx4)로 연산이 가능합니다.  

### 활성화함수(activation function)

앞서 비선형성을 위해 활성화함수를 사용한다고 하였습니다. 행렬연산을 보면 선형변환으로 이루어져 있는데요.  아무리 선형변환을 여러번 하더라도 결국 선형변환이 결과로 나옵니다. 예를 들어볼까요?  $f(x)=3x+2$, $g(x)=4x+1$ 을 연결하면 $f(g(x))=3(4x+1)+2=12x+5$ 가 됩니다. 결국 레이어를 아무리 많이 쌓아도 한 번 쌓은 것과 똑같습니다.  그렇기 때문에 비선형성을 추가하기 위해 활성화 함수를 사용합니다. 활성화함수에도 종류가 여러가지가 있지만 가장 유명한 **시그모이드 함수**를 살펴 보겠습니다. 

$\sigma(x)=\frac{1}{1+exp(-x)}$   

기대했던 것보다 너무 간단한가요?  $exp(x)$ 는 $e^x$ 입니다. 실제로 그래프를 그려보면 시그모이드 함수의 결과값은 **0과 1사이의 실수**로 나타난다는 것을 알 수 있습니다.  이미 눈치채신 분들도 있겠지만 시그모이드의 결과값이 확률처럼 쓰이기도 합니다.(로지스틱 회귀)  `numpy`, `matplotlib`으로 시그모이드 함수를 그래프에 나타내보겠습니다.  


```python
import numpy as np
import matplotlib.pyplot as plt 
```


```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```


```python
x = np.linspace(-8, 8, 100)
y = sigmoid(x)
plt.plot(x, y)
plt.show()
```


![output_13_0](https://user-images.githubusercontent.com/13113652/184499604-e127350d-b469-4a76-b738-d29bbe9c1086.png)


시그모이드 함수는 x가 0일 때 y는 0.5가 되고 0과 1사이에서 s자 곡선의 꼴을 나타내는 그래프 입니다.  이제 활성화함수 하나를 알았으니 실제로 코드로 직접 신경망을 만들어 보겠습니다.  

## 신경망 예측 실습

이번 실습을 통하여 *numpy* 를 통해 신경망의 예측을 구현해보겠습니다.  또한, 머신러닝 라이브러리로 많이 사용하는 *pytorch* 도 함께 구현하여 차이점을 알아보도록 하겠습니다.  참고로 저는 처음에 머신러닝을 공부할 때 *keras* 로 공부했었습니다.  어떤 라이브러리를 사용하든 정확히 알고만 사용하면 큰 차이는 없지만 개인적으로 *pytorch*가 가장 직관적이고 편리합니다.  신경망을 배우는 입장에서는 *numpy*로 layer를 직접 구현해보는 것이 확실히 이해하는데 도움이 되기 때문에 둘다 실습해보겠습니다.  


```python
class Sigmoid:
    def __init__(self):
        self.params = []
    
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
```

우선 위에서 보았던 활성화함수 `Sigmoid`를 클래스로 구현하였습니다.  `self.params` 는 학습 파라미터를 담고 있는 변수입니다.  시그모이드는 레이어의 결과값에 비선형선을 추가하는 역할을 하는 것이지 학습 파라미터를 업데이트 하지 않습니다.  따라서 해당 코드에서는 빈(empty) 리스트를 갖습니다.  


```python
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
    
    def forward(self, x):
        W, b = self.params
        return np.matmul(x, W) + b
```

다음으로 `Affine` 클래스 입니다.  아핀변환은 기하학에서의 변환을 뜻하는데요. 선형변환을 뜻한다고 생각하시면 됩니다.  학습 파라미터는 W(가중치)와 b(편향)을 가지며, 순전파가 일어나면 입력값 x와 행렬연산을 통해 결과값을 리턴합니다.  


```python
class NN_numpy:
    def __init__(self, input_size=2, hidden_size=4, output_size=3):
        W1 = np.random.randn(input_size, hidden_size)
        b1 = np.random.randn(hidden_size)
        W2 = np.random.randn(hidden_size, output_size)
        b2 = np.random.randn(output_size)
        
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2),
        ]
        
        self.params = []
        for layer in self.layers:
            self.params += layer.params
    
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
```

*numpy*로 구현한 아주 간단한 Neural Network 입니다.  각 아핀 변환에서의 학습 파라미터를 각각 W, b로 난수로 초기화 하였고  `self.layers`에 위에서의 `Affine`, `Sigmoid`를 추가하여 레이어를 쌓았습니다.  신경망 모델에서 `predict` 메소드를 호출할 때, 레이어의 `forward` 메소드를 통해 순전파가 발생합니다.  


```python
x = np.random.randn(10, 2)
model_numpy = NN_numpy(input_size=2, hidden_size=4, output_size=3)
out_numpy = model_numpy.predict(x)
out_numpy.shape
```




    (10, 3)



2차원 feature 를 가지는 10개의 샘플(x)를 신경망에 넣었을 때 3차원으로 변환된 결과가 나오는 것을 확인할 수 있습니다.  


```python
for param in model_numpy.params:
    print(param.shape)
    print(param)
    print()
```

    (2, 4)
    [[ 0.70943743  2.02464985 -1.43265056 -0.98758141]
     [-1.0456442  -0.92976807 -0.06964786  0.99213495]]
    
    (4,)
    [ 1.44433871 -1.11848055  1.04408346  1.61268503]
    
    (4, 3)
    [[-0.48803966  0.68726479 -0.75323276]
     [-0.14658621 -1.5626481  -0.35448567]
     [ 0.17685921  0.33741607  0.16612063]
     [-0.08993748  0.28330807 -2.00524662]]
    
    (3,)
    [-1.10223222 -0.7383462   1.12258491]



학습 파라미터를 보면 random 한 값들로 초기화가 되어 있습니다.  순전파만 발생했기 때문에 모델이 초기화될 때 랜덤한 값들로 구성이 된 이후로 업데이트 되지는 않았습니다.  학습 파라미터가 업데이트 되는 것은 다음 시간 *역전파*를 공부할 때 알아보도록 하겠습니다.  이번 시간의 키 포인트는 모델을 구성하는 학습 파라미터와 입력 데이터의 행렬연산 즉, 선형변환에 의해서 최종 결과가 도출된다는 것입니다.  

다음은 같은 신경망을 `pytorch`로 구현해보겠습니다.  


```python
import torch
import torch.nn as nn 
```


```python
class NN_torch(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, output_size=3):
        super().__init__()
        self.layer1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.layer2 = nn.Linear(in_features=hidden_size, out_features=output_size)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.sigmoid(x)
        x = self.layer2(x)
        return x
```


```python
x_torch = torch.tensor(x).float()
x_torch.shape
```




    torch.Size([10, 2])




```python
model_torch = NN_torch(input_size=2, hidden_size=4, output_size=3)
out_torch = model_torch(x_torch)
out_torch.shape
```




    torch.Size([10, 3])



`torch.nn` 에는 신경망 관련 라이브러리가 잘 구성되어 있기 때문에 `nn.Module`을 상속하여 클래스를 구현하면 됩니다.  `forward` 메소드는 `nn.Module` 에서 상속을 받았기 때문에 오버라이딩을 통해 구현하시면 되고  pytorch 의 경우 모델 객체를 함수 콜하듯이 `model_torch(x_torch)` 인풋값을 모델에 주입할 수 있습니다.  결과값의 shape 를 통해 신경망이 잘 동작함을 알 수 있습니다.  


```python
for param in model_torch.parameters():
    print(param.shape)
    print(param)
    print()
```

    torch.Size([4, 2])
    Parameter containing:
    tensor([[-0.2487, -0.1411],
            [-0.2317, -0.4321],
            [ 0.0768,  0.1541],
            [-0.5610, -0.5094]], requires_grad=True)
    
    torch.Size([4])
    Parameter containing:
    tensor([0.2022, 0.0937, 0.4571, 0.3154], requires_grad=True)
    
    torch.Size([3, 4])
    Parameter containing:
    tensor([[ 0.1982, -0.0740,  0.0631, -0.4177],
            [-0.0114,  0.4622, -0.1800, -0.3551],
            [-0.1526,  0.1708,  0.2700,  0.3530]], requires_grad=True)
    
    torch.Size([3])
    Parameter containing:
    tensor([ 0.3663, -0.1895, -0.3060], requires_grad=True)



torch 역시 랜덤한 값들로 학습 파라미터들이 초기화 되었습니다.  그런데 weight 를 보니 numpy 로 구현했을 때와 shape 가 바뀌어 있다는 것을 확인할 수 있는데요.  torch 내부 코드를 보면 $y=xW^T+b$ 다음과 같이 동작하도록 되어 있습니다.  따라서 첫 번째 weight 도 transpose 되어 [4,2] 형태가 아니라 [2,4] 형태로 연산이 됩니다.  결국 초기화된 랜덤한 값만 다르지 모델이 형태가 동일함을 알 수 있습니다.  

이번 시간을 통해 신경망이 데이터를 입력으로 받아 순전파 발생시 어떻게 예측이 발생하는 지를 알아보았습니다.  다음 시간에는 모델이 학습을 어떻게 하는 지, 역전파에서 무슨 일이 일어나는 지 알아보도록 하겠습니다.   
