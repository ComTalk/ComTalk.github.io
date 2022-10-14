---
layout: single  
title: 자연어 처리를 위한 신경망 (2) - 역전파
categories: "Neural_Net"
use_math: true  
author: goeon
toc: true
tock-sticky: true
---

지난 시간에는 신경망이 어떻게 예측하는지 `순전파`를 통해 알아보았습니다. 이번 시간에는 `순전파`를 바탕으로 어떻게 신경망이 학습하는지(가중치를 업데이트 하는지) 알아보겠습니다. 가장 중요한 개념이라고 할 수 있는 `경사하강법(gradient descent)`에 알아보겠습니다. `역전파`는 **미분이 다했다** 라고도 할 수 있을 것 같은데요. 이번 시간을 통해 정확하게 이해해보셨으면 좋겠습니다. 그럼 시작해보죠. 이번 포스팅도 [밑바닥부터 시작하는 딥러닝2](https://www.hanbit.co.kr/store/books/look.php?p_code=B8950212853)를 참조하여 작성 되었습니다.  

## 신경망에서 '학습'이란?

신경망은 *무언가*를 예측하기 위해 존재합니다. 그렇다면 신경망이 학습한다는 것은 *무언가*에 대한 예측력이 높아지는 것을 의미하겠죠. 그렇다면 예측력이 올라간다는 건 무엇을 의미할까요? 행렬곱을 이루는 가중치들이 실제 정답에 가까워지도록 변경된다는 것입니다. 예측력이 올라가도록 가중치가 업데이트 되기 위해선 정답과의 유사하도록 예측을 해야할것입니다. 신경망에서는 정답과의 차이를 손실함수를 통해 정의합니다. 신경망이 예측한 값과 실제 정답과의 차이를 손실함수를 통해 정의하고 이를 줄이는 방향으로 가중치를 업데이트 하는 것입니다.  

## 손실함수 

보통은 task에 따라 사용하는 손실함수가 정해져있습니다. 물론 손실함수를 직접 정의해서 사용할 수도 있죠. 예를 들어 강아지, 고양이, 햄스터 사진을 보여주면서 어떤 동물인지 맞추는 task가 있다고 해봅시다. 이러한 task를 다중 클래스 분류 문제라고 할 수 있을텐데, 보통 *소프트맥스(softmax)* 함수를 통해 확률로 표현하고, *크로스 엔트로피(cross entropy)* 를 손실함수로 정의합니다. 강아지는 [1,0,0], 고양이는 [0,1,0], 햄스터는 [0,0,1] 라고 정답이 주어진다고 했을 때 `softmax`, `cross-entropy` 가 무엇인지 설명드릴게요.

## 소프트맥스(Softmax)

*선수식후설명*. 제가 설명할 때 주로 이렇게 설명드릴겁니다. 딥러닝을 공부할 때 수식을 몰라도 충분히 할 수 있다고는 하지만, 정확하게 알아야만 확장을 할 수 있습니다. 정말 어려운 수식이 아니라면 꼭! 꼭! 이해하고 넘어가야 한다고 생각하는데요. 다행히 소프트맥스는 수식이 굉장히 쉽습니다. 

$y_k=\frac{exp(s_k)}{\sum_{i=1}^{n}{exp(s_i)}}$

$n=3$ 이라고 해보면 $y_1=\frac{exp(s_1)}{exp(s_1)+exp(s_2)+exp(s_3)}$ 으로 나타낼 수 있겠죠. 결국 0~1 사이의 값, 즉 확률로 나타낼 것입니다. 

그러면 그냥 $\frac{s_1}{s_1+s_2+s_3}$ 이라고 하면 될 것을 왜 굳이 지수함수를 사용할까요? 우선 미분가능하도록 만들기 위해서입니다. 그리고 큰 값은 확실히 크게, 작은 값이 확실히 작게 나타내어 클래스 분류를 더 잘하도록 만들기 위해 그렇습니다. 아래 지수함수 그래프를 보시면 x값이 클수록 y값의 변화는 훨씬 큰 것을 볼 수가 있습니다. 신경망이 예측한 값을 지수함수에 대입하여 그 차이를 극대화 하는 것이죠


```python
import numpy as np 
import matplotlib.pyplot as plt 

x = np.linspace(-5, 10, 1000)
y = np.exp(x)

plt.plot(x, y)
```

    
<img src="https://github.com/ComTalk/ComTalk.github.io/blob/master/assets/images/nlp_image/exponential_func.png?raw=true" width="35%" height="35%">
    


예를 들어, 강아지, 고양이, 햄스터에 대한 신경망의 예측값이 [3.1, -4.4, 2.8] 였다고 해봅시다. 그리고 softmax 를 취하면 [0.574, 0.001, 0.425] 이 될 것입니다.  

## 크로스 엔트로피(Cross Entropy)

$L=-\sum_{k}{t_klog{y_k}}=\sum_{k}{log\frac{1}{y_k} \cdot t_k}$

여기서 $t$는 [1, 0, 0]과 같은 정답 레이블을 뜻합니다. k개의 클래스를 나타내는 벡터입니다. 위에서 예측한 값이 실제 강아지라고 했을 때 cross entropy 를 구해보겠습니다. $log{\frac{1}{0.574}} \cdot 1+log{\frac{1}{0.001}} \cdot 0+log{\frac{1}{0.425}} \cdot 0=0.56$

만약 예측을 조금더 잘해서 softmax 값이 [0.8, 0.1, 0.1]로 나왔다면 크로스엔트로피는 더 낮게 나왔을 것입니다. $log{\frac{1}{0.8}} \cdot 1+log{\frac{1}{0.1}} \cdot 0+log{\frac{1}{0.1}} \cdot 0=0.22$  

이제 소프트맥스, 크로스엔트로피에 대해 아셨습니다. 결국 신경망의 학습은 손실함수를 0에 가까워지도록 가중치를 업데이트 한다는 것을 이해하셨을 겁니다. 그렇다면 마지막으로 가중치는 도대체 **어떻게** 업데이트 하는 것일까요??

## 미분과 기울기(gradient)

서두에 **역전파는 미분이 다했다**라고 했던 말 기억하시나요? 신경망 학습에서 가장 중요한 것은 그레디언트를 구하는 것입니다. $y=x^2$라는 식을 생각해봅시다. $y$를 $x$에 대해 미분하면 $\frac{\partial{y}}{\partial{x}}=2x$가 됩니다. 

그렇다면 이 미분값의 의미가 무엇일까요? $y=x^2$식에서 각 $x$에서의 기울기를 뜻합니다. 또한 $y$를 $x$에 대해 미분했기 때문에 $x$의 변화가 $y$에 미치는 영향이라고도 해석할 수 있습니다. 


```python
x = np.linspace(-10, 10, 1000)
y = x * x 
plt.plot(x, y, c='black')
plt.axline((0, -4), (2, 4), c='r')
plt.axline((0, -25), (5, 25), c='b')
plt.grid()
```


    
<img src="https://github.com/ComTalk/ComTalk.github.io/blob/master/assets/images/nlp_image/derivative.png?raw=true" width="35%" height="35%">
    


빨간선은 점(2, 4)에서의 기울기(2*2=4), 파란선은 점(5, 25)에서의 기울기(5*2=10)를 나타냅니다.

그럼 왜 미분값을 알아야 할까요? *얼마나 weight 를 업데이트 할 것인지* 에 대한 힌트를 제공하기 때문입니다. 

먼저 신경망 역전파의 핵심인 연쇄법칙(chain rule)에 대해 설명하고 넘어가겠습니다.

## 연쇄 법칙 (chain rule)

chain rule 이란 합성함수를 미분할 때 각 미분에 대한 곱으로 나타낼 수 있는 법칙입니다. 

$y=f(x)$, $z=g(y)$ 라는 함수가 있습니다. $z=g(f(x))$ 라고 쓸 수 있습니다. 이 때 x에 대해 미분하면 $\frac{\partial{z}}{\partial{x}}=\frac{\partial{z}}{\partial{y}} \cdot \frac{\partial{y}}{\partial{x}}$ 으로 나타낼 수 있습니다.

아래 그림을 보면 여러 계산 식을 계산 그래프로 나타내었습니다. $e=a+b$, $f=c+d$, $j=e \cdot f$ 와 같은 식이 있습니다. 이들의 최종결과값이 $L$ 이라고 한다면 각 매개변수가 $L$에 미치는 영향이 어느정도 되는지 알기 위해 $L$에 대하여 각각의 매개변수를 미분하면 빨간선과 같은 식이 됩니다. 빨간선의 방향이 파란선과 반대방향인 이유는 최종 결과값부터 입력층으로 반대로 흐르기 때문입니다. 네. 맞습니다. 이게 바로 역전파에요. 


<img src="https://github.com/ComTalk/ComTalk.github.io/blob/master/assets/images/nlp_image/bp1.png?raw=true" width="35%" height="35%">

그런데 이 각각의 계산식이 결국 층에 따라 합성함수의 꼴로 나타나는데요. 그렇기 때문에 아래와 같이 연쇄법칙을 이용해서 그래디언트를 계산할 수 있게 됩니다! 즉, layer가 얼마나 쌓여 있든지 간에 연쇄법칙을 이용하여 gradient 를 구할 수 있는 것이죠!

<img src="https://github.com/ComTalk/ComTalk.github.io/blob/master/assets/images/nlp_image/bp2.png?raw=true" width="35%" height="35%">

자, 이제 그럼 실제 신경망의 메인 연산인 행렬곱을 구현하면서 이해해보죠!

$y=xW$ 라는 식이 있습니다. 각각의 shape 을 보면 $x$는 (1xD) 벡터, $W$는 (DxH) 행렬, $y$는 (1xH) 벡터 입니다. $y$ 이후에도 어떠한 식이 있고 그 이후 최종 결과 스칼라값을 $L$이라고 하겠습니다.

$x$의 $i$번째 원소에 대한 미분은 $\frac{\partial{L}}{\partial{x_i}}=\sum_{j}{\frac {\partial{L}} {\partial{y_j}} {\frac{\partial{y_j}}{\partial{x_i}}}}$ 와 같이 됩니다. $x_i$를 조금 변화시켰을 때 $L$이 얼마나 변할 것인가를 뜻하겠죠. 

$x_i$ 가 조금 변하면 $y$를 구성하는 원소들도 다 변할 것이고 이에 따라 최종 결과값 $L$도 변할텐데요. $\frac{\partial{y_j}}{\partial{x_i}}=W_{ij}$ 인 것을 알기 때문에 결국  $\frac{\partial{L}}{\partial{x_i}}=\sum_{j}{\frac{\partial{L}}{\partial{y_j}}{\frac{\partial{y_j}}{\partial{x_i}}}}=\sum_{j}{\frac{\partial{L}}{\partial{y_j}}W_{ij}}$. 즉 $\frac{\partial{L}}{\partial{x}}=\frac{\partial{L}}{\partial{y}}W^{T}$ (T는 행과 열을 바꾼 전치행렬)입니다. shape 을 확인해 봐도 (1xD)=(1xH)x(HxD)로 성립함을 알 수 있죠. 

이제 실제로 구현해봅시다.


```python
class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
    
    def forward(self, x):
        W, = self.params
        out = np.matmul(x, W)
        self.x = x
        return out 

    def backward(self, dout):
        W, = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        self.grads[0][...] = dW # deep copy
        return dx
```

생략기호 `...`은 얕은 복사가 아닌 깊은 복사를 뜻합니다.

지난 시간 신경망 예측 시간에 구현했던 `Sigmoid`, `Affine` 클래스에도 `backward`를 추가해보겠습니다. 

시그모이드 함수 $y=\frac{1}{1+e^{-x}}$ 을 미분하면 $\frac{\partial{y}}{\partial{x}}=y(1-y)$ 입니다. 출력층에서 전해진 기울기 ($\frac{\partial{L}}{\partial{y}}$) 에 시그모이드 함수의 미분 ($\frac{\partial{y}}{\partial{x}}$)를 곱하고 그 값을 입력 쪽 계층으로 전파합니다. 


```python
class Sigmoid:
    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None 
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
```

MatMul 클래스처럼 Affine 클래스도 역전파를 구현하면 bias 가 추가되는 것 밖에 달라진 점이 없습니다. 


```python
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None 
    
    def forward(self, x):
        W, b = self.params
        out = np.matmul(x, W) + b
        self.x = x
        return out
    
    def backward(self, dout):
        W, b = self.params 
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        db = np.sum(dout, axis=0)
        
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx
```

학습 가중치가 어떻게 업데이트 되는지를 통해 신경망의 역전파에 대해 알 수 있었습니다. 오늘 내용이 조금 이해가 안 되더라도 차근차근 이해하는데 집중하시고요. 저는 시간이 지나면 자주 까먹더라구요. 이렇게 numpy 로 구현해보는게 많은 도움이 되었습니다. 

이제 다음시간 부터는 정말 자연어처리에 대해 다뤄보겠습니다. 오늘도 수고하셨습니다 :D
