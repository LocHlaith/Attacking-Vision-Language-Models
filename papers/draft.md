
---

# 第1章 问题叙述

## 1.1 研究背景与现实意义

随着人工智能技术的发展，视觉分类模型已经广泛应用于自动驾驶、无人机侦察、军事目标识别、智能安防、工业检测、人脸识别、遥感图像分析等场景。对于一个视觉分类系统而言，其基本任务是：给定一幅输入图像，模型自动判断图像所属类别。例如，在交通场景中，模型需要识别交通标志、行人、车辆和障碍物；在军事场景中，模型可能需要识别飞机、舰船、导弹发射装置或地面目标；在公共安全领域，模型可能需要识别危险物品、异常行为或敏感目标。

然而，现代视觉模型虽然在标准测试集上具有较高准确率，但其决策过程通常是由大量非线性运算组合而成，外部观察者难以直接解释其内部判别机制。因此，在实际应用中，视觉模型可能面临有意设计的输入干扰。所谓“攻击视觉模型”，在本文中并不是指破坏模型参数，而是指在输入图像上加入某种受到约束的微小扰动，使模型输出发生错误变化。

这种问题具有重要现实意义。

第一，从信息安全角度看，视觉模型已经成为许多智能系统的入口。如果输入图像经过微小修改后即可导致系统误判，则说明系统存在潜在安全风险。例如，智能门禁、安防监控、自动审核系统和身份识别系统都可能受到输入层面的干扰。

第二，从国家安全角度看，视觉识别系统被广泛应用于军事侦察、无人系统导航、战场态势感知和目标检测。如果敌方能够通过物理图案、伪装涂装或局部遮挡干扰识别系统，则可能影响军事装备的识别、跟踪和决策。

第三，从运筹学角度看，该问题本质上可以被抽象为一个带约束的非线性优化问题。其决策变量是图像像素扰动，目标函数是分类器的损失函数或分类置信度函数，约束条件是扰动幅度、图像取值范围、扰动面积或扰动形状。因此，攻击视觉模型不仅是人工智能安全问题，也可以作为传统运筹学中非线性规划、目标规划、罚函数法、对偶理论、数值优化和图论方法的综合应用案例。

本文选择 MobileNetV2 图像分类模型作为研究对象，尝试从运筹学角度建立“攻击 MobileNetV2”的数学模型，并分别使用传统解析法、传统数值法、对偶思想以及图论方法进行建模与算法设计。

---

## 1.2 MobileNetV2 分类模型介绍

MobileNetV2 是一种轻量级卷积神经网络，主要用于图像分类任务。它的设计目标是在保持较好分类精度的同时降低计算量和参数量，因此常被用于移动端、嵌入式设备和资源受限场景。TorchVision 官方文档说明，MobileNetV2 的模型构建器可以使用不同的预训练权重，其中包括 ImageNet-1K 权重；PyTorch Hub 也说明，预训练视觉模型通常期望输入为三通道 RGB 图像，并进行标准归一化处理。([PyTorch Documentation][1])

在本文中，设输入图像为

[
x\in \mathbb{R}^{H\times W\times C}.
]

对于常用 ImageNet 输入尺寸，取

[
H=224,\quad W=224,\quad C=3.
]

因此输入可展开为一个向量：

[
x=(x_1,x_2,\ldots,x_n)^\top\in\mathbb{R}^n,
]

其中

[
n=224\times 224\times 3=150528.
]

每个 (x_i) 表示图像中的一个颜色通道像素值。例如，左上角像素的红色通道、绿色通道、蓝色通道分别可以对应三个不同的决策变量。

MobileNetV2 是一个多类别分类函数，可表示为

[
F:\mathbb{R}^{150528}\rightarrow \mathbb{R}^{K}.
]

在 ImageNet-1K 分类任务中，

[
K=1000.
]

因此，模型输出为

[
F(x)=
\begin{bmatrix}
z_1(x)\
z_2(x)\
\vdots\
z_{1000}(x)
\end{bmatrix}.
]

其中 (z_i(x)) 称为第 (i) 类的 logit，也可理解为第 (i) 类的未归一化得分函数。MobileNetV2 的 TorchVision 源码实现中，分类器部分包含线性层，输出类别数由 `num_classes` 控制；使用 ImageNet 权重时对应 ImageNet-1K 分类任务。([PyTorch Documentation][2])

对于输入图像 (x)，分类器首先计算所有类别的得分：

[
z_1(x),z_2(x),\ldots,z_K(x).
]

然后通过最大得分规则得到预测类别：

[
\hat y(x)=\arg\max_{1\le j\le K}z_j(x).
]

若真实类别为 (y)，则分类正确的条件为

[
\hat y(x)=y.
]

若经过扰动后的图像为

[
x'=x+\delta,
]

则攻击成功的条件为

[
\hat y(x+\delta)\ne y.
]

---

## 1.3 MobileNetV2 的得分函数与损失函数

MobileNetV2 对输入图像的处理过程可以抽象为一系列函数复合。设模型共有 (L) 层，则可以写为

[
F(x)=F_L\circ F_{L-1}\circ \cdots \circ F_2\circ F_1(x).
]

其中，各层可能包括卷积、批归一化、ReLU6 激活、深度可分离卷积、倒残差结构、全局平均池化和线性分类层。MobileNetV2 的核心结构包括 inverted residual blocks 和 linear bottlenecks，这是其区别于传统卷积网络的重要设计。([PyTorch Documentation][3])

因此，每一个 logit 都可以看成输入图像像素的非线性函数：

[
z_j(x)=z_j(x_1,x_2,\ldots,x_{150528}),
\quad j=1,2,\ldots,1000.
]

为了将分类问题转化为可微优化问题，通常引入 softmax 概率：

[
p_j(x)
======

\frac{e^{z_j(x)}}{\sum_{k=1}^{K}e^{z_k(x)}}.
]

其中

[
p_j(x)
]

可理解为模型将输入图像 (x) 判断为第 (j) 类的归一化概率。

对于真实类别 (y)，定义交叉熵损失函数：

[
J(x,y)
======

-\log p_y(x).
]

将 softmax 展开，可得

[
J(x,y)
======

-\log
\frac{e^{z_y(x)}}{\sum_{j=1}^{K}e^{z_j(x)}}.
]

进一步化简：

[
J(x,y)
======

-\left[
z_y(x)-\log\left(\sum_{j=1}^{K}e^{z_j(x)}\right)
\right],
]

即

[
J(x,y)
======

## \log\left(\sum_{j=1}^{K}e^{z_j(x)}\right)

z_y(x).
]

当模型对真实类别 (y) 越自信时，(p_y(x)) 越大，(J(x,y)) 越小；当模型对真实类别 (y) 越不自信时，(p_y(x)) 越小，(J(x,y)) 越大。

因此，从攻击角度看，目标并不是减小 (J(x,y))，而是希望增大 (J(x,y))。为了保持传统运筹学“最小化问题”的统一形式，本文定义攻击目标函数

[
\Phi(x,y)=-J(x,y).
]

于是，攻击问题可以写成最小化

[
\Phi(x+\delta,y)
================

-J(x+\delta,y).
]

当 (\Phi) 越小，等价于 (J) 越大，模型对真实类别的判断越不稳定。

---

# 第2章 解析法求解带二范数约束的非线性规划模型

## 2.1 算法设计：非线性规划建模与传统解析法

本章将“攻击 MobileNetV2”建模为一个带二范数扰动约束的非线性规划问题，并分别讨论 Lagrange 函数法、Fritz John 条件和 Kuhn-Tucker 条件。

---

### 2.1.1 原始攻击模型

设原始图像为

[
x\in\mathbb{R}^n,
\quad n=150528.
]

设扰动向量为

[
\delta\in\mathbb{R}^n.
]

扰动后图像为

[
x'=x+\delta.
]

为了保证扰动不能过大，引入二范数平方约束：

[
|\delta|_2^2\le \varepsilon^2.
]

其中

[
|\delta|_2^2
============

\sum_{i=1}^{n}\delta_i^2.
]

此外，由于图像像素通常需要限制在合法范围内，若像素归一化到 ([0,1])，则还应满足盒约束：

[
0\le x_i+\delta_i\le 1,
\quad i=1,2,\ldots,n.
]

为了突出主模型，本节首先研究二范数平方约束，盒约束可在后续数值实现中通过投影或附加不等式处理。

由于攻击目标是使真实类别 (y) 的交叉熵损失增大，而传统优化一般采用最小化形式，定义

[
\Phi(\delta)
============

-J(x+\delta,y).
]

其中

[
J(x+\delta,y)
=============

-\log
\frac{e^{z_y(x+\delta)}}{\sum_{j=1}^{K}e^{z_j(x+\delta)}}.
]

因此

[
\Phi(\delta)
============

\log
\frac{e^{z_y(x+\delta)}}{\sum_{j=1}^{K}e^{z_j(x+\delta)}}.
]

进一步化简：

[
\Phi(\delta)
============

## z_y(x+\delta)

\log
\left(
\sum_{j=1}^{K}e^{z_j(x+\delta)}
\right).
]

于是，攻击模型可写为：

[
\begin{aligned}
\min_{\delta\in\mathbb{R}^n}
\quad
&
\Phi(\delta)
============

## z_y(x+\delta)

\log
\left(
\sum_{j=1}^{K}e^{z_j(x+\delta)}
\right)\
s.t.
\quad
&
|\delta|_2^2\le \varepsilon^2.
\end{aligned}
]

记约束函数为

[
g(\delta)=|\delta|_2^2-\varepsilon^2.
]

则约束为

[
g(\delta)\le 0.
]

因此模型为标准不等式约束非线性规划：

[
\begin{aligned}
\min_{\delta}
\quad & \Phi(\delta)\
s.t.
\quad & g(\delta)\le 0.
\end{aligned}
]

---

### 2.1.2 二范数平方约束的光滑性

二范数平方为

[
|\delta|_2^2
============

\delta^\top\delta.
]

其梯度为

[
\nabla_\delta |\delta|_2^2
==========================

2\delta.
]

其 Hessian 矩阵为

[
\nabla_\delta^2 |\delta|_2^2
============================

2I.
]

因此，二范数平方是处处可导、处处二阶可导的光滑凸函数。

约束函数

[
g(\delta)=\delta^\top\delta-\varepsilon^2
]

的梯度为

[
\nabla g(\delta)=2\delta.
]

Hessian 为

[
\nabla^2 g(\delta)=2I.
]

因此约束集合

[
{\delta:|\delta|_2^2\le \varepsilon^2}
]

是一个闭球，是凸集。

然而，目标函数 (\Phi(\delta)) 是由 MobileNetV2 的 logit 函数复合而成。由于神经网络包含多层非线性变换，(\Phi(\delta)) 一般是非凸函数。因此，整个问题属于：

[
\text{凸约束 + 非凸目标}
]

的非线性规划问题。

---

## 2.1.3 Lagrange 函数法

对于问题

[
\begin{aligned}
\min_{\delta}
\quad & \Phi(\delta)\
s.t.
\quad & g(\delta)\le 0,
\end{aligned}
]

其中

[
g(\delta)=\delta^\top\delta-\varepsilon^2.
]

构造 Lagrange 函数：

[
\mathcal{L}(\delta,\lambda)
===========================

\Phi(\delta)
+
\lambda g(\delta).
]

即

[
\mathcal{L}(\delta,\lambda)
===========================

\Phi(\delta)
+
\lambda(\delta^\top\delta-\varepsilon^2).
]

其中 (\lambda) 为 Lagrange 乘子。

如果最优解位于约束边界上，即

[
|\delta|_2^2=\varepsilon^2,
]

则可按等式约束处理，令

[
h(\delta)=\delta^\top\delta-\varepsilon^2=0.
]

此时 Lagrange 必要条件为：

[
\nabla_\delta \mathcal{L}(\delta,\lambda)=0,
]

即

[
\nabla \Phi(\delta)+\lambda \nabla h(\delta)=0.
]

由于

[
\nabla h(\delta)=2\delta,
]

可得

[
\nabla \Phi(\delta)+2\lambda\delta=0.
]

于是

[
\nabla \Phi(\delta)=-2\lambda\delta.
]

若

[
\lambda\ne 0,
]

则

[
\delta
======

-\frac{1}{2\lambda}\nabla\Phi(\delta).
]

这说明，在边界最优点处，扰动方向与目标函数梯度方向共线。

由于

[
\Phi(\delta)=-J(x+\delta,y),
]

所以

[
\nabla \Phi(\delta)
===================

-\nabla_\delta J(x+\delta,y).
]

于是

[
-\nabla_\delta J(x+\delta,y)+2\lambda\delta=0.
]

即

[
2\lambda\delta
==============

\nabla_\delta J(x+\delta,y).
]

因此

[
\delta
======

\frac{1}{2\lambda}
\nabla_\delta J(x+\delta,y).
]

这个式子说明：为了增大损失函数 (J)，扰动方向应当与损失函数关于输入的梯度方向相关。

但由于 (\nabla_\delta J(x+\delta,y)) 本身依赖于 (\delta)，这是一个隐式方程。一般不能直接解析求出全局最优解。

因此，可进一步采用一阶泰勒近似。

在 (\delta=0) 附近展开：

[
J(x+\delta,y)
\approx
J(x,y)+\nabla_xJ(x,y)^\top\delta.
]

于是

[
\Phi(\delta)
============

-J(x+\delta,y)
\approx
-J(x,y)-\nabla_xJ(x,y)^\top\delta.
]

由于 (-J(x,y)) 与 (\delta) 无关，可忽略常数项，近似问题为：

[
\begin{aligned}
\min_{\delta}
\quad
&
-\nabla_xJ(x,y)^\top\delta\
s.t.
\quad
&
|\delta|_2^2\le \varepsilon^2.
\end{aligned}
]

等价于：

[
\begin{aligned}
\max_{\delta}
\quad
&
\nabla_xJ(x,y)^\top\delta\
s.t.
\quad
&
|\delta|_2\le \varepsilon.
\end{aligned}
]

根据 Cauchy-Schwarz 不等式：

[
\nabla_xJ(x,y)^\top\delta
\le
|\nabla_xJ(x,y)|_2|\delta|_2.
]

又因为

[
|\delta|_2\le \varepsilon,
]

所以

[
\nabla_xJ(x,y)^\top\delta
\le
\varepsilon |\nabla_xJ(x,y)|_2.
]

当

[
\delta
======

\varepsilon
\frac{\nabla_xJ(x,y)}{|\nabla_xJ(x,y)|_2}
]

时等号成立。

因此，基于一阶泰勒近似和 Lagrange 思想，可得到解析扰动方向：

[
\delta^{(1)}
============

\varepsilon
\frac{\nabla_xJ(x,y)}{|\nabla_xJ(x,y)|_2}.
]

该解不是原非凸问题的全局最优解，而是一阶近似模型下的全局最优解。对于原问题而言，它只能被视为局部近似解或满意解。

---

## 2.1.4 Fritz John 条件

Fritz John 条件适用于更一般的约束优化问题。对于问题

[
\begin{aligned}
\min_{\delta}
\quad & \Phi(\delta)\
s.t.
\quad & g(\delta)\le 0,
\end{aligned}
]

Fritz John 条件指出：若 (\delta^*) 是局部最优解，则存在不全为零的乘子

[
\lambda_0\ge 0,\quad \lambda_1\ge 0,
]

使得

[
\lambda_0\nabla\Phi(\delta^*)
+
\lambda_1\nabla g(\delta^*)
===========================

0,
]

同时满足

[
\lambda_1g(\delta^*)=0,
]

[
g(\delta^*)\le 0,
]

[
\lambda_0,\lambda_1\ge 0.
]

代入

[
g(\delta)=\delta^\top\delta-\varepsilon^2,
]

可得

[
\lambda_0\nabla\Phi(\delta^*)
+
2\lambda_1\delta^*
==================

0.

]

如果

[
\lambda_0>0,
]

则两边除以 (\lambda_0)，令

[
\lambda=\frac{\lambda_1}{\lambda_0},
]

得到

[
\nabla\Phi(\delta^*)+2\lambda\delta^*=0.
]

这就退化为 KKT 条件中的驻点方程。

但 Fritz John 条件允许

[
\lambda_0=0.
]

如果

[
\lambda_0=0,
]

则条件变为

[
2\lambda_1\delta^*=0.
]

由于乘子不全为零，若 (\lambda_1>0)，则

[
\delta^*=0.
]

这说明 Fritz John 条件比 KKT 条件更弱。在本问题中，由于约束集合

[
|\delta|_2^2\le \varepsilon^2
]

通常满足一定的约束资格条件，所以更常用的是 KKT 条件。

因此，Fritz John 条件可用于说明局部最优解必须满足的广义必要条件，但它不能直接给出解析最优扰动，也不能保证全局最优。其结论性质为：

[
\text{局部最优必要条件，而非充分条件。}
]

---

## 2.1.5 Kuhn-Tucker 条件

对于标准不等式约束问题

[
\begin{aligned}
\min_{\delta}
\quad & \Phi(\delta)\
s.t.
\quad & g(\delta)\le 0,
\end{aligned}
]

其中

[
g(\delta)=\delta^\top\delta-\varepsilon^2.
]

Kuhn-Tucker 条件为：

第一，原始可行性：

[
g(\delta^*)\le 0.
]

即

[
|\delta^*|_2^2\le \varepsilon^2.
]

第二，对偶可行性：

[
\lambda^*\ge 0.
]

第三，互补松弛条件：

[
\lambda^*g(\delta^*)=0.
]

即

[
\lambda^*(|\delta^*|_2^2-\varepsilon^2)=0.
]

第四，驻点条件：

[
\nabla\Phi(\delta^*)+\lambda^*\nabla g(\delta^*)=0.
]

由于

[
\nabla g(\delta^*)=2\delta^*,
]

所以

[
\nabla\Phi(\delta^*)+2\lambda^*\delta^*=0.
]

又因为

[
\Phi(\delta)=-J(x+\delta,y),
]

所以

[
-\nabla_\delta J(x+\delta^*,y)+2\lambda^*\delta^*=0.
]

因此

[
2\lambda^*\delta^*
==================

\nabla_\delta J(x+\delta^*,y).
]

若约束不起作用，则

[
g(\delta^*)<0.
]

由互补松弛条件可得

[
\lambda^*=0.
]

此时驻点条件变为

[
\nabla\Phi(\delta^*)=0.
]

这表示最优解是目标函数在球内部的驻点。

但对于攻击问题，一般希望在给定扰动预算内尽量增大损失，因此最优解通常位于边界：

[
|\delta^*|_2^2=\varepsilon^2.
]

此时

[
\lambda^*>0.
]

KKT 条件给出：

[
\delta^*
========

\frac{1}{2\lambda^*}
\nabla_\delta J(x+\delta^*,y).
]

同样，这是一个隐式方程。由于目标函数非凸，KKT 条件一般只是局部最优的必要条件，不保证全局最优。

如果对目标函数进行一阶泰勒展开，则可以得到上一节中的闭式满意解：

[
\delta^{(1)}
============

\varepsilon
\frac{\nabla_xJ(x,y)}{|\nabla_xJ(x,y)|_2}.
]

若进行二阶泰勒展开，则有：

[
J(x+\delta,y)
\approx
J(x,y)+g^\top\delta+\frac12\delta^\top H\delta,
]

其中

[
g=\nabla_xJ(x,y),
]

[
H=\nabla_x^2J(x,y).
]

于是近似攻击问题变为：

[
\begin{aligned}
\max_{\delta}
\quad
&
g^\top\delta+\frac12\delta^\top H\delta\
s.t.
\quad
&
\delta^\top\delta\le\varepsilon^2.
\end{aligned}
]

等价地，写成最小化形式：

[
\begin{aligned}
\min_{\delta}
\quad
&
-g^\top\delta-\frac12\delta^\top H\delta\
s.t.
\quad
&
\delta^\top\delta\le\varepsilon^2.
\end{aligned}
]

构造 Lagrange 函数：

[
\mathcal{L}(\delta,\lambda)
===========================

-g^\top\delta-\frac12\delta^\top H\delta
+
\lambda(\delta^\top\delta-\varepsilon^2).
]

驻点条件为：

[
-g-H\delta+2\lambda\delta=0.
]

即

[
(2\lambda I-H)\delta=g.
]

若矩阵

[
2\lambda I-H
]

可逆，则

[
\delta(\lambda)
===============

(2\lambda I-H)^{-1}g.
]

再由边界条件

[
\delta(\lambda)^\top\delta(\lambda)=\varepsilon^2
]

求出 (\lambda)。

该二阶近似模型比一阶模型更精细，但需要 Hessian 矩阵，计算代价极高。因此它更适合作为理论推导，而不一定适合作为大规模实际求解方法。二阶近似模型的解是局部二次模型下的候选最优解，对于原始神经网络非凸问题，一般只能视作局部解或满意解。

---

## 2.1.6 解析法所得解的性质分析

本节所用三种解析法的结论如下。

Lagrange 函数法适合处理等式约束或活跃不等式约束。对于本文问题，它能推导出最优扰动与梯度方向的关系：

[
\nabla\Phi(\delta^*)+2\lambda\delta^*=0.
]

但由于 (\Phi) 是复杂非凸函数，该方程一般不能解析求全局解。

Fritz John 条件是最一般的局部最优必要条件。它不要求严格的约束资格条件，但结论较弱，只能说明局部最优点处梯度之间存在某种线性相关关系。

Kuhn-Tucker 条件在满足约束资格条件时更常用。它给出了原始可行性、对偶可行性、互补松弛和驻点条件，但对于非凸问题一般只是必要条件，而不是充分条件。

因此，本章解析法的结论可概括为：

[
\text{原问题：非凸非线性规划，解析法一般只能给出局部最优必要条件。}
]

若使用一阶泰勒展开，则得到近似问题的全局最优解：

[
\delta^{(1)}
============

\varepsilon
\frac{\nabla_xJ(x,y)}{|\nabla_xJ(x,y)|_2}.
]

但它对于原始问题只是满意解。

若使用二阶泰勒展开，则可得到更复杂的信赖域型子问题：

[
(2\lambda I-H)\delta=g.
]

该解是局部二次模型下的候选解，仍不能保证原问题全局最优。

---

# 第3章 数值法求解带二范数约束的非线性规划模型

## 3.1 算法设计：传统数值优化方法

本章继续研究如下约束非线性规划问题：

[
\begin{aligned}
\min_{\delta}
\quad
&
\Phi(\delta)
============

-J(x+\delta,y)\
s.t.
\quad
&
|\delta|_2^2\le \varepsilon^2.
\end{aligned}
]

为了便于数值求解，也可以采用目标规划或罚函数形式，将约束并入目标函数。

---

## 3.1.1 罚函数与目标规划形式

定义二范数平方扰动：

[
D(\delta)=|\delta|_2^2.
]

若希望同时控制扰动大小并提高攻击成功程度，可构造目标规划模型：

[
\min_{\delta}
\quad
D(\delta)-P J(x+\delta,y).
]

即

[
\min_{\delta}
\quad
|\delta|_2^2-PJ(x+\delta,y).
]

其中 (P>0) 是权重系数。

当 (P) 较大时，模型更重视增大损失函数；当 (P) 较小时，模型更重视减小扰动规模。

也可引入攻击成功阈值 (\eta)，建立约束模型：

[
\begin{aligned}
\min_{\delta}
\quad
&
|\delta|_2^2\
s.t.
\quad
&
J(x+\delta,y)\ge \eta.
\end{aligned}
]

其中 (\eta) 表示希望损失函数至少达到的水平。由于约束

[
J(x+\delta,y)\ge \eta
]

不好直接处理，可转化为违反量惩罚：

[
\max{\eta-J(x+\delta,y),0}.
]

于是目标规划模型为：

[
\min_{\delta}
\quad
|\delta|_2^2
+
P\max{\eta-J(x+\delta,y),0}.
]

如果满足

[
J(x+\delta,y)\ge \eta,
]

则惩罚项为零；否则惩罚项为正。

为了提高光滑性，可将

[
\max{u,0}
]

替换为光滑近似函数，例如：

[
\psi_\alpha(u)
==============

\frac{1}{\alpha}\log(1+e^{\alpha u}),
]

其中 (\alpha>0)。当 (\alpha) 较大时，

[
\psi_\alpha(u)\approx \max{u,0}.
]

于是光滑罚函数模型为：

[
\min_{\delta}
\quad
|\delta|*2^2
+
P\psi*\alpha(\eta-J(x+\delta,y)).
]

这使得传统数值优化方法更容易使用。

---

## 3.1.2 最速下降法

对于无约束罚函数模型，设

[
Q(\delta)
=========

|\delta|_2^2-PJ(x+\delta,y).
]

目标为

[
\min_\delta Q(\delta).
]

最速下降法的基本迭代格式为：

[
\delta^{(k+1)}
==============

## \delta^{(k)}

\alpha_k\nabla Q(\delta^{(k)}),
]

其中 (\alpha_k>0) 为步长。

首先计算梯度：

[
\nabla Q(\delta)
================

## \nabla |\delta|_2^2

P\nabla_\delta J(x+\delta,y).
]

由于

[
\nabla |\delta|_2^2=2\delta,
]

因此

[
\nabla Q(\delta)
================

## 2\delta

P\nabla_\delta J(x+\delta,y).
]

所以迭代公式为：

[
\delta^{(k+1)}
==============

## \delta^{(k)}

\alpha_k
\left[
2\delta^{(k)}
-------------

P\nabla_\delta J(x+\delta^{(k)},y)
\right].
]

即

[
\delta^{(k+1)}
==============

(1-2\alpha_k)\delta^{(k)}
+
\alpha_k P\nabla_\delta J(x+\delta^{(k)},y).
]

若保留二范数约束

[
|\delta|_2\le\varepsilon,
]

则每次迭代后需要进行投影：

[
\delta^{(k+1)}
==============

\Pi_{\mathcal{B}_2(\varepsilon)}
\left[
\delta^{(k)}
------------

\alpha_k\nabla \Phi(\delta^{(k)})
\right],
]

其中

[
\mathcal{B}_2(\varepsilon)
==========================

{\delta:|\delta|_2\le\varepsilon}.
]

二范数球上的投影为：

[
\Pi_{\mathcal{B}_2(\varepsilon)}(v)
===================================

\begin{cases}
v, & |v|_2\le\varepsilon,[4pt]
\varepsilon\frac{v}{|v|_2}, & |v|_2>\varepsilon.
\end{cases}
]

由于

[
\Phi(\delta)=-J(x+\delta,y),
]

有

[
\nabla\Phi(\delta)
==================

-\nabla_\delta J(x+\delta,y).
]

因此投影最速下降法可写为：

[
\delta^{(k+1)}
==============

\Pi_{\mathcal{B}*2(\varepsilon)}
\left[
\delta^{(k)}
+
\alpha_k\nabla*\delta J(x+\delta^{(k)},y)
\right].
]

该方法每一步沿着使损失 (J) 增大的方向移动，然后将结果投影回允许扰动范围。

最速下降法计算简单，适合高维问题，但收敛速度可能较慢。由于目标函数非凸，若算法收敛，一般只能保证得到局部驻点或满意解，而不能保证全局最优。

---

## 3.1.3 牛顿法

对于无约束模型

[
\min_\delta Q(\delta),
]

牛顿法基于二阶泰勒展开。

在当前点 (\delta^{(k)}) 附近，有

[
Q(\delta^{(k)}+s)
\approx
Q(\delta^{(k)})
+
\nabla Q(\delta^{(k)})^\top s
+
\frac12 s^\top \nabla^2Q(\delta^{(k)})s.
]

记

[
g_k=\nabla Q(\delta^{(k)}),
]

[
H_k=\nabla^2Q(\delta^{(k)}).
]

则二阶近似模型为：

[
m_k(s)
======

Q(\delta^{(k)})
+
g_k^\top s
+
\frac12s^\top H_ks.
]

牛顿方向 (s_k) 由最小化该二次模型得到：

[
\nabla_s m_k(s)=g_k+H_ks=0.
]

因此

[
H_ks_k=-g_k.
]

若 (H_k) 可逆，则

[
s_k=-H_k^{-1}g_k.
]

迭代公式为：

[
\delta^{(k+1)}
==============

\delta^{(k)}+\alpha_ks_k.
]

其中 (\alpha_k) 可通过线搜索确定。

在本文模型中，

[
Q(\delta)=|\delta|_2^2-PJ(x+\delta,y).
]

因此

[
\nabla Q(\delta)=2\delta-P\nabla_\delta J(x+\delta,y),
]

[
\nabla^2Q(\delta)=2I-P\nabla_\delta^2J(x+\delta,y).
]

所以牛顿方程为：

[
\left[
2I-P\nabla_\delta^2J(x+\delta^{(k)},y)
\right]s_k
==========

-\left[
2\delta^{(k)}-P\nabla_\delta J(x+\delta^{(k)},y)
\right].
]

即

[
\left[
2I-P\nabla_\delta^2J(x+\delta^{(k)},y)
\right]s_k
==========

## P\nabla_\delta J(x+\delta^{(k)},y)

2\delta^{(k)}.
]

由于输入维度

[
n=150528,
]

Hessian 矩阵规模为

[
150528\times150528.
]

完整存储和求逆代价极高。因此，牛顿法在理论上可用于二阶光滑近似模型，但直接用于完整 MobileNetV2 输入优化时计算困难。

此外，MobileNetV2 中包含 ReLU6 等分段线性激活函数，目标函数在某些点可能非二阶光滑。因此牛顿法的适用性通常依赖于局部光滑近似。

牛顿法若在局部区域内 Hessian 正定，且初始点足够接近局部极小点，可具有较快局部收敛速度。但对于非凸问题，牛顿法不能保证全局最优，可能收敛到局部驻点、鞍点或数值上的满意解。

---

## 3.1.4 拟牛顿法

拟牛顿法的思想是不直接计算 Hessian，而是用矩阵 (B_k) 近似 Hessian：

[
B_k\approx \nabla^2Q(\delta^{(k)}).
]

搜索方向由

[
B_ks_k=-\nabla Q(\delta^{(k)})
]

确定。

即

[
s_k=-B_k^{-1}\nabla Q(\delta^{(k)}).
]

迭代为：

[
\delta^{(k+1)}
==============

\delta^{(k)}+\alpha_ks_k.
]

设

[
p_k=\delta^{(k+1)}-\delta^{(k)},
]

[
q_k=\nabla Q(\delta^{(k+1)})-\nabla Q(\delta^{(k)}).
]

拟牛顿法要求 Hessian 近似满足割线条件：

[
B_{k+1}p_k=q_k.
]

常见更新形式之一为 BFGS 更新：

[
B_{k+1}
=======

## B_k

\frac{B_kp_kp_k^\top B_k}{p_k^\top B_kp_k}
+
\frac{q_kq_k^\top}{q_k^\top p_k}.
]

由于 (n=150528) 很大，完整 (B_k) 矩阵仍然难以存储。因此实际可使用有限记忆思想，只保留最近若干组

[
(p_k,q_k)
]

来近似计算搜索方向。

拟牛顿法比牛顿法计算负担小，不需要显式 Hessian，但仍依赖目标函数的局部光滑性和较合理的线搜索。对于本文非凸模型，拟牛顿法可作为求满意解的数值方法，一般不能保证全局最优。

若加入二范数约束，则可采用投影拟牛顿思想：

[
\tilde{\delta}^{(k+1)}
======================

\delta^{(k)}+\alpha_ks_k,
]

[
\delta^{(k+1)}
==============

\Pi_{\mathcal{B}_2(\varepsilon)}
(\tilde{\delta}^{(k+1)}).
]

其中

[
\Pi_{\mathcal{B}_2(\varepsilon)}
]

为二范数球投影算子。

---

## 3.1.5 共轭梯度法

共轭梯度法最初用于求解正定二次优化问题：

[
\min_\delta
\frac12\delta^\top A\delta-b^\top\delta,
\quad A\succ0.
]

其核心思想是构造一组关于 (A) 共轭的搜索方向：

[
d_i^\top A d_j=0,\quad i\ne j.
]

对于本文问题，可以在当前点附近对目标函数进行二阶泰勒展开：

[
Q(\delta^{(k)}+s)
\approx
Q(\delta^{(k)})
+
g_k^\top s
+
\frac12s^\top H_ks.
]

若在该局部二次模型中使用共轭梯度法，则需要求解：

[
\min_s
\quad
g_k^\top s+\frac12s^\top H_ks.
]

其一阶条件为：

[
H_ks=-g_k.
]

共轭梯度法可以在不显式求逆 (H_k) 的情况下迭代求解该线性方程组。

在非线性优化中，也可直接使用非线性共轭梯度法。设

[
g_k=\nabla Q(\delta^{(k)}).
]

初始方向为

[
d_0=-g_0.
]

之后迭代方向为：

[
d_k=-g_k+\beta_kd_{k-1}.
]

其中 (\beta_k) 可取 Fletcher-Reeves 形式：

[
\beta_k^{FR}
============

\frac{g_k^\top g_k}{g_{k-1}^\top g_{k-1}}.
]

也可取 Polak-Ribiere 形式：

[
\beta_k^{PR}
============

\frac{g_k^\top(g_k-g_{k-1})}{g_{k-1}^\top g_{k-1}}.
]

迭代为：

[
\delta^{(k+1)}
==============

\delta^{(k)}+\alpha_kd_k.
]

若保留约束，则使用投影：

[
\delta^{(k+1)}
==============

\Pi_{\mathcal{B}_2(\varepsilon)}
\left(
\delta^{(k)}+\alpha_kd_k
\right).
]

共轭梯度法不需要存储完整 Hessian，适合大规模问题。但它在严格意义上最适合凸二次模型；对于本文这样的非凸非线性模型，通常只能作为启发式数值方法或局部二次近似方法。所得结果一般是局部驻点或满意解。

---

## 3.1.6 数值法所得解的性质分析

最速下降法只使用一阶梯度，计算简单，适合高维输入，但可能收敛较慢。对非凸问题，一般得到局部解或满意解。

牛顿法使用二阶信息，局部收敛速度快，但需要 Hessian 矩阵，计算代价极高，并且要求较强的局部光滑性。对于 MobileNetV2 这类分段非线性模型，牛顿法更适合作为理论分析方法，而不是直接大规模求解方法。

拟牛顿法避免直接计算 Hessian，通过迭代近似二阶信息，在计算量和收敛速度之间折中。对于本文问题，可作为较实用的传统数值优化方法，但不保证全局最优。

共轭梯度法适合大规模问题，尤其适合局部二次近似模型。对于非凸神经网络目标，它可以用于求满意解，但不能保证全局最优。

因此，本章数值法的总体结论为：

[
\text{传统数值法可以用于构造扰动，但由于目标非凸，通常得到满意解或局部最优解。}
]

---

# 第4章 基于对偶思想的攻击模型

## 4.1 算法设计：由损失约束构造对偶问题

第2章和第3章主要采用形式：

[
\begin{aligned}
\min_{\delta}
\quad
&
-J(x+\delta,y)\
s.t.
\quad
&
|\delta|_2^2\le \varepsilon^2.
\end{aligned}
]

该模型表示：在扰动预算不超过 (\varepsilon) 的情况下，尽可能增大分类损失。

本章采用另一种建模思路：把攻击成功程度作为约束，把扰动大小作为目标。该形式更接近传统运筹学中的“资源最小化问题”。

---

## 4.1.1 原始问题：最小扰动模型

设希望损失函数至少达到阈值 (\eta)，即

[
J(x+\delta,y)\ge \eta.
]

其中 (\eta) 是预先设定的攻击成功水平。若 (\eta) 足够大，则说明模型对真实类别 (y) 的分类信心明显下降。

于是建立原始问题：

[
\begin{aligned}
\min_{\delta}
\quad
&
|\delta|_2^2\
s.t.
\quad
&
J(x+\delta,y)\ge \eta.
\end{aligned}
]

为了写成标准小于等于约束，定义

[
c(\delta)=\eta-J(x+\delta,y).
]

约束

[
J(x+\delta,y)\ge \eta
]

等价于

[
c(\delta)\le 0.
]

因此原始问题为：

[
\begin{aligned}
\min_{\delta}
\quad
&
D(\delta)=|\delta|_2^2\
s.t.
\quad
&
c(\delta)=\eta-J(x+\delta,y)\le 0.
\end{aligned}
]

---

## 4.1.2 Lagrange 函数

构造 Lagrange 函数：

[
\mathcal{L}(\delta,\lambda)
===========================

D(\delta)+\lambda c(\delta).
]

即

[
\mathcal{L}(\delta,\lambda)
===========================

|\delta|_2^2
+
\lambda[\eta-J(x+\delta,y)].
]

其中

[
\lambda\ge 0.
]

展开：

[
\mathcal{L}(\delta,\lambda)
===========================

\delta^\top\delta
+
\lambda\eta
-----------

\lambda J(x+\delta,y).
]

对于固定的 (\lambda)，需要求解

[
\inf_{\delta}\mathcal{L}(\delta,\lambda).
]

定义对偶函数：

[
q(\lambda)
==========

\inf_{\delta}
\left[
\delta^\top\delta
+
\lambda\eta
-----------

\lambda J(x+\delta,y)
\right].
]

由于

[
\lambda\eta
]

与 (\delta) 无关，可写为

[
q(\lambda)
==========

\lambda\eta
+
\inf_{\delta}
\left[
\delta^\top\delta
-----------------

\lambda J(x+\delta,y)
\right].
]

对偶问题为：

[
\max_{\lambda\ge 0}q(\lambda).
]

即

[
\max_{\lambda\ge 0}
\left{
\lambda\eta
+
\inf_{\delta}
\left[
|\delta|_2^2
------------

\lambda J(x+\delta,y)
\right]
\right}.
]

这说明：对偶问题可以被理解为一系列带权目标规划问题。对于给定的 (\lambda)，求解

[
\min_{\delta}
\quad
|\delta|_2^2
------------

\lambda J(x+\delta,y).
]

然后再调整 (\lambda)，使攻击成功约束与扰动大小之间达到平衡。

---

## 4.1.3 KKT 条件

原始问题为：

[
\begin{aligned}
\min_{\delta}
\quad
&
|\delta|_2^2\
s.t.
\quad
&
\eta-J(x+\delta,y)\le 0.
\end{aligned}
]

KKT 条件如下。

第一，原始可行性：

[
\eta-J(x+\delta^*,y)\le 0.
]

即

[
J(x+\delta^*,y)\ge \eta.
]

第二，对偶可行性：

[
\lambda^*\ge 0.
]

第三，互补松弛：

[
\lambda^*
[
\eta-J(x+\delta^*,y)
]
=0.
]

第四，驻点条件：

[
\nabla_\delta
\left[
|\delta|*2^2
+
\lambda(\eta-J(x+\delta,y))
\right]*{\delta=\delta^*}
=0.
]

由于

[
\nabla_\delta|\delta|_2^2=2\delta,
]

且

[
\nabla_\delta[\eta-J(x+\delta,y)]
=================================

-\nabla_\delta J(x+\delta,y),
]

所以驻点条件为：

[
2\delta^*
---------

# \lambda^*\nabla_\delta J(x+\delta^*,y)

0.

]

因此

[
2\delta^*
=========

\lambda^*
\nabla_\delta J(x+\delta^*,y).
]

即

[
\delta^*
========

\frac{\lambda^*}{2}
\nabla_\delta J(x+\delta^*,y).
]

若攻击成功约束正好活跃，则

[
J(x+\delta^*,y)=\eta.
]

此时

[
\lambda^*>0.
]

若约束不活跃，即

[
J(x+\delta^*,y)>\eta,
]

则由互补松弛可得

[
\lambda^*=0.
]

此时驻点条件给出

[
\delta^*=0,
]

但这通常与

[
J(x,y)\ge\eta
]

相关。若原始图像本身已经满足攻击成功阈值，则无需扰动；否则最优解一般位于约束边界上。

---

## 4.1.4 对偶问题的数值求解

由于

[
q(\lambda)
==========

\inf_{\delta}
\left[
|\delta|_2^2
+
\lambda\eta
-----------

\lambda J(x+\delta,y)
\right],
]

可采用双层迭代：

内层：对固定 (\lambda)，求

[
\delta(\lambda)
===============

\arg\min_\delta
\left[
|\delta|_2^2
------------

\lambda J(x+\delta,y)
\right].
]

外层：更新 (\lambda\ge0)，使约束满足。

内层目标函数为

[
Q_\lambda(\delta)
=================

## |\delta|_2^2

\lambda J(x+\delta,y).
]

其梯度为

[
\nabla Q_\lambda(\delta)
========================

2\delta-\lambda\nabla_\delta J(x+\delta,y).
]

可使用最速下降法：

[
\delta^{(k+1)}
==============

## \delta^{(k)}

\alpha_k
[
2\delta^{(k)}
-------------

\lambda\nabla_\delta J(x+\delta^{(k)},y)
].
]

外层可根据约束违反程度更新 (\lambda)。若

[
J(x+\delta(\lambda),y)<\eta,
]

说明攻击约束尚未满足，应增大 (\lambda)；若

[
J(x+\delta(\lambda),y)>\eta
]

且扰动过大，则可适当减小 (\lambda)。

一种简单更新为：

[
\lambda^{(t+1)}
===============

\max\left{
0,\lambda^{(t)}
+\rho_t[\eta-J(x+\delta^{(t)},y)]
\right}.
]

其中 (\rho_t>0) 是外层步长。

该方法属于传统 Lagrange 乘子或罚函数思想下的数值求解方法。

---

## 4.1.5 对偶方法的解的性质

如果原问题是凸优化问题，并满足一定约束资格条件，则强对偶可能成立，原问题最优值等于对偶问题最优值。

但本文问题中，虽然

[
|\delta|_2^2
]

是凸函数，约束函数中包含

[
J(x+\delta,y),
]

而 (J) 由 MobileNetV2 这种深层非线性模型决定，一般是非凸函数。因此原问题不是凸优化问题。

所以，对偶方法不能保证无对偶间隙，也不能保证全局最优。

本章所得结论为：

[
\text{对偶模型可以提供目标规划解释和乘子更新机制，但一般只能求局部解或满意解。}
]

从运筹学角度看，本章的价值在于：它把“攻击成功程度”转化为约束，把“扰动大小”转化为资源消耗，再利用 Lagrange 乘子解释两者之间的权衡关系。

---

# 第5章 基于图论的结构化扰动模型：最小“张天羽”图案

## 5.1 算法设计：从像素扰动到图论优化

前几章研究的扰动主要是连续扰动，即每个像素通道都可以发生微小变化。本章研究一种特殊的结构化扰动：

> 被扰动的像素全部置为黑色，并且这些被置黑的像素在图像上组成“张天羽”三个字。目标是在攻击成功的前提下，使“张天羽”三个字的面积最小，即被置黑像素数量最少。

该问题同时具有图像结构约束、离散选择约束和模型攻击约束，因此可以从图论和组合优化角度建模。

---

## 5.1.1 基本变量定义

设图像共有 (m=H\times W) 个像素位置。对于 (224\times224) 图像，

[
m=224\times224=50176.
]

每个像素位置记为

[
v_i,\quad i=1,2,\ldots,m.
]

将图像像素位置看作图的节点集合：

[
V={v_1,v_2,\ldots,v_m}.
]

若两个像素在空间上相邻，例如上下左右相邻，则在它们之间连一条边：

[
E={(v_i,v_j):v_i\text{ 与 }v_j\text{ 相邻}}.
]

于是得到图像网格图：

[
G=(V,E).
]

定义二元变量：

[
s_i=
\begin{cases}
1, & \text{像素 }v_i\text{ 被置黑},\
0, & \text{像素 }v_i\text{ 不被扰动}.
\end{cases}
]

所有被置黑的像素集合为

[
S={v_i\in V:s_i=1}.
]

其面积为

[
|S|=\sum_{i=1}^{m}s_i.
]

目标是使

[
|S|
]

尽可能小。

---

## 5.1.2 “置黑扰动”的数学表达

设原图像为

[
x\in[0,1]^{H\times W\times 3}.
]

若像素位置 (v_i) 被置黑，则该位置三个通道均变为 0：

[
x'*{i,R}=0,\quad x'*{i,G}=0,\quad x'_{i,B}=0.
]

若不被扰动，则保持原值：

[
x'*{i,c}=x*{i,c},
\quad c\in{R,G,B}.
]

因此可写为：

[
x'*{i,c}=(1-s_i)x*{i,c}.
]

因为当 (s_i=0) 时，

[
x'*{i,c}=x*{i,c};
]

当 (s_i=1) 时，

[
x'_{i,c}=0.
]

于是扰动后图像为

[
x'(s)=x\odot(1-s),
]

其中 (s) 在空间维度上对 RGB 三个通道复制，(\odot) 表示逐元素乘法。

攻击成功条件为：

[
\hat y(x'(s))\ne y.
]

等价地，可用 logit margin 表示：

[
\max_{k\ne y}z_k(x'(s))-z_y(x'(s))\ge 0.
]

因此，最小黑色像素攻击问题可写为：

[
\begin{aligned}
\min_s
\quad
&
\sum_{i=1}^{m}s_i\
s.t.
\quad
&
\max_{k\ne y}z_k(x'(s))-z_y(x'(s))\ge 0,\
&
s_i\in{0,1},
\quad i=1,2,\ldots,m,\
&
S\text{ 构成“张天羽”三个字的图案}.
\end{aligned}
]

这是一个带神经网络非线性约束的 0-1 组合优化问题。

---

## 5.1.3 “张天羽”字形约束

为了使被扰动像素组成“张天羽”三个字，需要引入字形模板。

设标准字形模板为

[
T\subseteq V.
]

其中 (T) 表示在某个标准尺寸、标准位置、标准字体下，“张天羽”三个字对应的像素集合。

如果允许缩放，则引入尺度参数 (a)。如果允许平移，则引入平移参数 (b=(b_1,b_2))。如果允许旋转，则引入旋转参数 (\theta)。

于是模板变换后的位置集合为：

[
T(a,b,\theta)
=============

{R_\theta(av)+b:v\in T}.
]

其中 (R_\theta) 为旋转矩阵。

为了简化，可先固定字体、位置和方向，只允许缩放。此时模板集合为

[
T(a).
]

若要求所有被置黑像素必须属于该模板，则有

[
S\subseteq T(a).
]

若要求形成完整可识别文字，则不能任意删除笔画，需要引入连通性和笔画完整性约束。

设“张天羽”三个字由若干笔画组成：

[
\mathcal{P}={P_1,P_2,\ldots,P_r}.
]

其中每个 (P_l\subseteq T(a)) 是一个笔画像素集合。

为了保持字形可识别，可要求每个笔画至少保留一定比例：

[
\sum_{i:v_i\in P_l}s_i
\ge
\gamma_l |P_l|,
\quad l=1,2,\ldots,r.
]

其中

[
0<\gamma_l\le 1.
]

若希望每个笔画连通，还需要对每个笔画子图

[
G[P_l\cap S]
]

施加连通性约束。

连通性可用图论方式表示。对于某个笔画 (P_l)，选择一个根节点 (r_l)，要求从根节点到所有被选节点存在路径。可通过流变量建模：

[
f_{ij}^{(l)}\ge 0,
\quad (i,j)\in E.
]

若节点 (v_i) 属于笔画 (P_l)，并且被选中，则需要有单位流到达。该类约束可写为网络流形式，但会显著增加模型规模。

因此，在实际算法中，可先固定完整模板，只改变尺度 (a)。此时问题简化为：

[
\min_a |T(a)|
\quad
s.t.
\quad
\hat y(x'(T(a)))\ne y.
]

即寻找面积最小的“张天羽”模板，使攻击成功。

---

## 5.1.4 图论抽象

将每个像素视为图节点，像素邻接关系视为边。于是“张天羽”图案可看作图像网格图中的一个特殊子图：

[
G_T=(T,E_T),
]

其中

[
E_T={(u,v)\in E:u\in T,v\in T}.
]

若选择部分像素置黑，则对应选择子图

[
G_S=(S,E_S).
]

目标是：

[
\min |V(G_S)|
]

使得

[
G_S
]

满足字形结构约束，并且

[
\hat y(x'(S))\ne y.
]

因此，该问题可以抽象为：

[
\text{在图 }G\text{ 中寻找满足攻击约束的最小字形子图。}
]

若把每个节点 (v_i) 对攻击目标的贡献定义为权重 (w_i)，例如

[
w_i
===

\left|
\frac{\partial J(x,y)}{\partial x_{i,R}}
\right|
+
\left|
\frac{\partial J(x,y)}{\partial x_{i,G}}
\right|
+
\left|
\frac{\partial J(x,y)}{\partial x_{i,B}}
\right|,
]

则 (w_i) 表示该像素对损失函数的敏感程度。

希望在“张天羽”字形约束下选择总贡献较大的像素，同时面积较小。于是可建立加权图优化模型：

[
\max_{S\subseteq T(a)}
\quad
\sum_{v_i\in S}w_i
]

[
s.t.
\quad
|S|\le M,
]

[
G_S\text{ 满足字形连通性约束}.
]

然后逐步减小或增大 (M)，寻找使攻击成功的最小面积。

---

## 5.1.5 数学难点一：非凸性

该问题的第一个难点是非凸性。

攻击成功约束为：

[
\max_{k\ne y}z_k(x'(s))-z_y(x'(s))\ge 0.
]

其中

[
z_k(x'(s))
]

由 MobileNetV2 计算得到，是关于 (s) 的复杂非线性函数。由于

[
x'*{i,c}=(1-s_i)x*{i,c},
]

所以

[
z_k(x'(s))
==========

z_k(x\odot(1-s)).
]

神经网络包含多层非线性变换，因此

[
z_k(x\odot(1-s))
]

一般不是凸函数，也不是凹函数。

而约束中又含有

[
\max_{k\ne y}
]

运算，使得整体更加复杂。

因此可行域

[
\left{
s:
\max_{k\ne y}z_k(x'(s))-z_y(x'(s))\ge 0
\right}
]

通常不是凸集。

这意味着传统凸优化理论不能直接保证全局最优解。

---

## 5.1.6 数学难点二：NP-hard

该问题的第二个难点是 NP-hard 性。

即使暂时忽略神经网络的复杂非线性，只考虑如下简化问题：

[
\min_s\sum_i s_i
]

[
s.t.
\quad
\sum_i w_i s_i\ge W,
]

[
s_i\in{0,1}.
]

该问题已经类似最小基数覆盖问题。若再加入笔画连通性、模板结构、多个区域覆盖要求，则可归约为集合覆盖、最小连通子图、Steiner tree 等经典 NP-hard 问题。

在原问题中，还存在神经网络攻击成功约束：

[
\hat y(x'(s))\ne y.
]

该约束并非简单线性约束，而是复杂黑箱或白箱函数约束。因此，原问题至少具有组合优化问题的复杂性。

所以可以判断：

[
\text{最小“张天羽”攻击问题一般是 NP-hard 的。}
]

这意味着不存在已知多项式时间算法能够对所有输入都精确求得全局最优解。因此，实际求解通常采用启发式算法、贪心算法、混合整数规划松弛或局部搜索方法。

---

## 5.1.7 数学难点三：可扩展性

对于 (224\times224) 图像，像素位置数量为

[
m=50176.
]

如果对每个像素设置一个 0-1 变量，则有

[
50176
]

个二元变量。

若对 RGB 通道分别建模，则变量数为

[
150528.
]

如果进一步引入边变量、连通性流变量、笔画约束和模板缩放变量，则变量和约束数量会快速增长。

例如，网格图中每个像素最多与 4 个邻居相连，边数近似为

[
|E|\approx 2H(W-1)+2W(H-1).
]

当

[
H=W=224
]

时，

[
|E|\approx 2\cdot224\cdot223+2\cdot224\cdot223
==============================================

199808.

]

若对每条边设置流变量，模型规模会进一步扩大。

因此，直接精确求解完整 MIP 模型可能非常困难。必须使用降维、模板固定、候选点筛选、贪心策略或局部优化。

---

## 5.1.8 方法一：Saliency map + 贪心选择

第一种可行方法是“敏感度图 + 贪心选择”。

首先计算每个像素的重要性权重：

[
w_i
===

\sum_{c\in{R,G,B}}
\left|
\frac{\partial J(x,y)}{\partial x_{i,c}}
\right|.
]

若将像素置黑，则变化量为

[
\Delta x_{i,c}
==============

# 0-x_{i,c}

-x_{i,c}.
]

利用一阶近似：

[
J(x+\Delta x,y)
\approx
J(x,y)+\nabla_xJ(x,y)^\top\Delta x.
]

单个像素 (i) 被置黑所造成的近似损失变化为：

[
\Delta J_i
\approx
\sum_{c\in{R,G,B}}
\frac{\partial J(x,y)}{\partial x_{i,c}}
(-x_{i,c}).
]

如果目标是增大 (J)，则希望选择使 (\Delta J_i) 较大的像素。

因此定义像素收益：

[
r_i
===

\sum_{c\in{R,G,B}}
\left[
-\frac{\partial J(x,y)}{\partial x_{i,c}}x_{i,c}
\right].
]

若

[
r_i>0,
]

表示将该像素置黑预计会增大损失。

在“张天羽”模板内，只选择收益较大的像素。

算法步骤如下。

第一步，确定候选模板集合：

[
T(a_1),T(a_2),\ldots,T(a_L).
]

其中 (a_l) 表示不同字号或面积。

第二步，对于每个模板 (T(a_l))，计算其中所有像素的收益 (r_i)。

第三步，按收益从大到小排序：

[
r_{i_1}\ge r_{i_2}\ge \cdots.
]

第四步，从空集开始逐步加入像素：

[
S_t={i_1,i_2,\ldots,i_t}.
]

第五步，每加入一批像素，就构造扰动图像

[
x'(S_t)
]

并检查：

[
\hat y(x'(S_t))\ne y.
]

第六步，第一次攻击成功时，记录面积：

[
|S_t|=t.
]

该方法计算简单，适合大规模图像。但由于每次只根据局部敏感度选择像素，可能忽略像素之间的组合效应。因此它一般得到满意解，而不能保证全局最优。

---

## 5.1.9 方法二：混合整数规划 MIP

第二种方法是建立混合整数规划模型。

定义二元变量：

[
s_i\in{0,1},
\quad i=1,\ldots,m.
]

目标函数为：

[
\min \sum_{i=1}^{m}s_i.
]

置黑图像为：

[
x'*{i,c}=(1-s_i)x*{i,c}.
]

攻击成功约束为：

[
\max_{k\ne y}z_k(x')-z_y(x')\ge 0.
]

但该约束高度非线性，不能直接放入普通 MIP。为此可以使用一阶线性化。

在原图像 (x) 附近，对 logit margin 函数进行线性展开。

定义 margin 函数：

[
M(x)
====

\max_{k\ne y}z_k(x)-z_y(x).
]

攻击成功条件为：

[
M(x')\ge 0.
]

在 (x) 附近一阶近似：

[
M(x')
\approx
M(x)+\nabla_xM(x)^\top(x'-x).
]

由于

[
x'*{i,c}-x*{i,c}
================

-s_ix_{i,c},
]

所以

[
\nabla_xM(x)^\top(x'-x)
=======================

\sum_{i=1}^{m}
\sum_{c\in{R,G,B}}
\frac{\partial M(x)}{\partial x_{i,c}}
(-s_ix_{i,c}).
]

定义

[
a_i
===

-\sum_{c\in{R,G,B}}
\frac{\partial M(x)}{\partial x_{i,c}}x_{i,c}.
]

则

[
M(x')
\approx
M(x)+\sum_{i=1}^{m}a_is_i.
]

攻击成功近似约束为：

[
M(x)+\sum_{i=1}^{m}a_is_i\ge 0.
]

即

[
\sum_{i=1}^{m}a_is_i\ge -M(x).
]

若限制像素必须来自“张天羽”模板 (T)，则：

[
s_i=0,\quad i\notin T.
]

于是 MIP 模型为：

[
\begin{aligned}
\min_s
\quad
&
\sum_{i=1}^{m}s_i\
s.t.
\quad
&
\sum_{i=1}^{m}a_is_i\ge -M(x),\
&
s_i=0,\quad i\notin T,\
&
s_i\in{0,1},
\quad i=1,\ldots,m.
\end{aligned}
]

如果加入笔画比例约束：

[
\sum_{i:v_i\in P_l}s_i
\ge
\gamma_l|P_l|,
\quad l=1,\ldots,r.
]

如果加入连通性约束，可使用网络流变量。对每个笔画 (P_l)，设根节点为 (r_l)，令

[
u_{ij}^{(l)}
]

表示边 ((i,j)) 上的流量。对于每个非根节点 (i\in P_l)，要求：

[
\sum_{j:(j,i)\in E}u_{ji}^{(l)}
-------------------------------

# \sum_{j:(i,j)\in E}u_{ij}^{(l)}

s_i.
]

对根节点要求发出总流量：

[
\sum_{j:(r_l,j)\in E}u_{r_lj}^{(l)}
-----------------------------------

# \sum_{j:(j,r_l)\in E}u_{jr_l}^{(l)}

\sum_{i\in P_l,i\ne r_l}s_i.
]

同时，为了保证只有被选中的节点才能通过流量，可加入大 (M) 约束：

[
u_{ij}^{(l)}\le M s_i,
]

[
u_{ij}^{(l)}\le M s_j.
]

这样可保证选择的笔画像素具有连通性。

MIP 方法比贪心法更接近全局组合优化，但由于攻击约束经过线性化，求得的是线性近似模型下的最优解。对于原始神经网络模型，它仍然是满意解或局部近似解。

---

## 5.1.10 方法三：图论启发式求解

第三种方法是图论启发式。

首先构造图：

[
G=(V,E).
]

对每个节点赋予权重 (r_i)，表示置黑该像素对攻击目标的贡献。

若希望在面积最小的前提下达到攻击目标，可将问题近似为：

[
\min |S|
]

[
s.t.
\quad
\sum_{i\in S}r_i\ge R,
]

[
G[S]\text{ 满足字形结构约束}.
]

其中 (R) 是攻击所需的近似贡献阈值。

该问题类似图上的最小权覆盖或最小连通子图问题。

一种启发式算法为：

第一步，生成多个不同尺寸的“张天羽”模板子图：

[
G_{T(a_1)},G_{T(a_2)},\ldots,G_{T(a_L)}.
]

第二步，对每个模板计算总收益：

[
R(a_l)=\sum_{i\in T(a_l)}r_i.
]

第三步，按面积从小到大排序模板：

[
|T(a_1)|\le |T(a_2)|\le \cdots\le |T(a_L)|.
]

第四步，从面积最小模板开始测试攻击是否成功。

第五步，若某个模板成功，则尝试删除低收益节点，但保持字形连通性。删除规则为：

[
i^*
===

\arg\min_{i\in S}r_i.
]

若删除 (i^*) 后仍满足字形结构并攻击成功，则接受删除：

[
S\leftarrow S\setminus{i^*}.
]

否则保留。

第六步，重复直到不能继续删除。

该方法可以看成“从完整字形开始的反向贪心剪枝”。其优点是始终保持“张天羽”图案的整体结构，缺点是可能陷入局部最优。

---

## 5.1.11 图论模型的解的性质分析

对于最小“张天羽”扰动问题，若完整求解原始模型：

[
\begin{aligned}
\min_s
\quad
&
\sum_i s_i\
s.t.
\quad
&
\hat y(x'(s))\ne y,\
&
s_i\in{0,1},\
&
S\text{ 构成“张天羽”图案},
\end{aligned}
]

则这是一个非凸、离散、组合优化问题，通常无法有效求得全局最优。

Saliency map + 贪心选择方法速度快、可扩展性好，但只使用局部梯度信息，因此得到的是满意解。

MIP 方法可以在近似线性模型下求全局最优，但由于神经网络攻击约束被线性化，所以对原始问题仍然只是近似最优。

图论启发式方法能够较好保留字形结构，适合处理“张天羽”这种形状约束，但由于采用贪心剪枝或局部搜索，一般得到局部最优或满意解。

因此，本章结论为：

[
\text{最小“张天羽”攻击问题适合用图论和组合优化建模，但实际求解通常依赖启发式方法。}
]

---
