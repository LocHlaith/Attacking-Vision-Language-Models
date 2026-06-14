# 第2-6章实验结果

说明：下列图片占位均按“原图、攻击图、扰动图”从左到右排列。原图标题中的数值按原始 Top-1 输出值书写；攻击图标题中的数值为实验记录中的 `decision_function`。可视化图片采用 `manual` 后端输出，两个后端的成功率、耗时和扰动规模统一放入对比表。

## 2.5 实验结果

### 2.5.1 实验图片与代码后端说明

本实验统一使用 `datasets` 中的四张图片。跑车与波斯猫这两张图片的共同特征是主体明显、单一，适合检验二范数约束扰动在清晰单目标图像上的攻击效果。雏菊图片的主体同样明显，但花朵数量较多，图像内部存在多个重复显著区域，适合观察扰动在多主体场景中的稳定性。温莎结领带图片则更接近“人类难以达成唯一分类共识”的情形：它既可能被理解为服饰、领带、人物局部，也可能被模型归入其他视觉模式，因此其分类结果更依赖模型自身的决策边界。

代码层面保留了两套正式后端：`manual` 后端强调与课件和报告公式逐项对应，便于展示 Lagrange、KKT、下降法、罚函数、对偶和网络流等运筹学过程；`efficient` 后端强调运行效率和可复现实验统计，便于完成大规模重复实验。两套后端共享同一批原图、目标函数、阈值定义和 CSV 输出规范，因而可作为同一数学模型在不同求解实现下的对照。

四张原图的基线决策如下。

| 图片 | 原图标题 | 原始决策函数值 |
|---|---|---:|
| `datasets/1.png` | 原图：sports car<br>14.517% | 0.8022 |
| `datasets/2.png` | 原图：Persian cat<br>9.729% | 1.6965 |
| `datasets/3.png` | 原图：Windsor tie<br>12.536% | 0.2301 |
| `datasets/4.png` | 原图：daisy<br>13.113% | 0.8445 |

### 2.5.2 一阶 Lagrange 满意解

**图 2-1 一阶 Lagrange 满意解攻击结果**

| 原图 | 攻击图 | 扰动图 |
|---|---|---|
| `datasets/1.png`<br>原图：sports car<br>14.517% | `outputs/final_square_validation/analytic_first_order/manual/1/attacked.png`<br>攻击图：convertible<br>7.4182 | `outputs/final_square_validation/analytic_first_order/manual/1/perturbation.png`<br>扰动图<br>L2=3.8236 |
| `datasets/2.png`<br>原图：Persian cat<br>9.729% | `outputs/final_square_validation/analytic_first_order/manual/2/attacked.png`<br>攻击图：paper towel<br>4.8748 | `outputs/final_square_validation/analytic_first_order/manual/2/perturbation.png`<br>扰动图<br>L2=4.0000 |
| `datasets/3.png`<br>原图：Windsor tie<br>12.536% | `outputs/final_square_validation/analytic_first_order/manual/3/attacked.png`<br>攻击图：groom<br>7.9366 | `outputs/final_square_validation/analytic_first_order/manual/3/perturbation.png`<br>扰动图<br>L2=3.9995 |
| `datasets/4.png`<br>原图：daisy<br>13.113% | `outputs/final_square_validation/analytic_first_order/manual/4/attacked.png`<br>攻击图：handkerchief<br>12.6193 | `outputs/final_square_validation/analytic_first_order/manual/4/perturbation.png`<br>扰动图<br>L2=4.0000 |

### 2.5.3 二阶 KKT 近似解

**图 2-2 二阶 KKT 近似攻击结果**

| 原图 | 攻击图 | 扰动图 |
|---|---|---|
| `datasets/1.png`<br>原图：sports car<br>14.517% | `outputs/final_square_validation/second_order_kkt/manual/1/attacked.png`<br>攻击图：convertible<br>7.2838 | `outputs/final_square_validation/second_order_kkt/manual/1/perturbation.png`<br>扰动图<br>L2=3.8161 |
| `datasets/2.png`<br>原图：Persian cat<br>9.729% | `outputs/final_square_validation/second_order_kkt/manual/2/attacked.png`<br>攻击图：paper towel<br>4.6844 | `outputs/final_square_validation/second_order_kkt/manual/2/perturbation.png`<br>扰动图<br>L2=3.9458 |
| `datasets/3.png`<br>原图：Windsor tie<br>12.536% | `outputs/final_square_validation/second_order_kkt/manual/3/attacked.png`<br>攻击图：groom<br>8.1068 | `outputs/final_square_validation/second_order_kkt/manual/3/perturbation.png`<br>扰动图<br>L2=3.9313 |
| `datasets/4.png`<br>原图：daisy<br>13.113% | `outputs/final_square_validation/second_order_kkt/manual/4/attacked.png`<br>攻击图：handkerchief<br>13.0017 | `outputs/final_square_validation/second_order_kkt/manual/4/perturbation.png`<br>扰动图<br>L2=4.0000 |

### 2.5.4 对比与分析

| 方法 | 后端 | 成功率 | 平均耗时/s | 最大耗时/s | 平均决策函数值 | 平均 L2 |
|---|---|---:|---:|---:|---:|---:|
| 一阶 Lagrange 满意解 | manual | 4/4 | 0.079 | 0.201 | 8.212 | 3.956 |
| 一阶 Lagrange 满意解 | efficient | 4/4 | 0.070 | 0.164 | 8.212 | 3.956 |
| 二阶 KKT 近似解 | manual | 4/4 | 56.442 | 69.630 | 8.269 | 3.923 |
| 二阶 KKT 近似解 | efficient | 4/4 | 55.922 | 66.698 | 8.269 | 3.923 |

两类解析近似方法均达到 100% 成功率，说明在二范数半径为 4 的可行域内，四张图像的原始决策点均离模型分类边界较近。从运筹学角度看，该问题具有凸的二范数球可行域，但目标函数来自神经网络复合映射，整体并非凸规划，因此实验结果更应理解为局部近似模型给出的满意解，而不是原非凸问题的全局最优证明。

一阶 Lagrange 满意解几乎把波斯猫、温莎结和雏菊三张图的扰动推到半径边界，体现出二范数约束在 KKT 语言中的“起作用约束”特征：当内部驻点不足以改变分类时，最有效的可行方向往往位于球面边界。二阶 KKT 近似在前三张图上平均 L2 略小，说明 Hessian 信息能够修正纯梯度方向，使候选点更贴近局部曲率下的边界；但其耗时约为一阶方法的 700 倍，计算代价主要来自二阶近似和线性方程求解。对课程中的“必要条件”而言，二阶方法更接近局部最优性检验；对实际实验而言，一阶方法已经以极低成本给出稳定可行解。

## 3.3 实验结果

### 3.3.1 加权和模型：最速下降法

**图 3-1 加权和模型的最速下降攻击结果**

| 原图 | 攻击图 | 扰动图 |
|---|---|---|
| `datasets/1.png`<br>原图：sports car<br>14.517% | `outputs/final_square_validation/weighted_steepest/manual/1/attacked.png`<br>攻击图：pickup<br>26.3531 | `outputs/final_square_validation/weighted_steepest/manual/1/perturbation.png`<br>扰动图<br>L2=7.1967 |
| `datasets/2.png`<br>原图：Persian cat<br>9.729% | `outputs/final_square_validation/weighted_steepest/manual/2/attacked.png`<br>攻击图：paper towel<br>44.0947 | `outputs/final_square_validation/weighted_steepest/manual/2/perturbation.png`<br>扰动图<br>L2=11.8668 |
| `datasets/3.png`<br>原图：Windsor tie<br>12.536% | `outputs/final_square_validation/weighted_steepest/manual/3/attacked.png`<br>攻击图：hog<br>48.5944 | `outputs/final_square_validation/weighted_steepest/manual/3/perturbation.png`<br>扰动图<br>L2=9.5552 |
| `datasets/4.png`<br>原图：daisy<br>13.113% | `outputs/final_square_validation/weighted_steepest/manual/4/attacked.png`<br>攻击图：handkerchief<br>37.5079 | `outputs/final_square_validation/weighted_steepest/manual/4/perturbation.png`<br>扰动图<br>L2=10.2925 |

### 3.3.2 加权和模型：牛顿法

**图 3-2 加权和模型的牛顿法攻击结果**

| 原图 | 攻击图 | 扰动图 |
|---|---|---|
| `datasets/1.png`<br>原图：sports car<br>14.517% | `outputs/final_square_validation/weighted_newton/manual/1/attacked.png`<br>攻击图：pickup<br>26.3531 | `outputs/final_square_validation/weighted_newton/manual/1/perturbation.png`<br>扰动图<br>L2=7.1967 |
| `datasets/2.png`<br>原图：Persian cat<br>9.729% | `outputs/final_square_validation/weighted_newton/manual/2/attacked.png`<br>攻击图：paper towel<br>37.3853 | `outputs/final_square_validation/weighted_newton/manual/2/perturbation.png`<br>扰动图<br>L2=12.6849 |
| `datasets/3.png`<br>原图：Windsor tie<br>12.536% | `outputs/final_square_validation/weighted_newton/manual/3/attacked.png`<br>攻击图：hog<br>48.5944 | `outputs/final_square_validation/weighted_newton/manual/3/perturbation.png`<br>扰动图<br>L2=9.5552 |
| `datasets/4.png`<br>原图：daisy<br>13.113% | `outputs/final_square_validation/weighted_newton/manual/4/attacked.png`<br>攻击图：handkerchief<br>37.3499 | `outputs/final_square_validation/weighted_newton/manual/4/perturbation.png`<br>扰动图<br>L2=9.4913 |

### 3.3.3 加权和模型：Levenberg-Marquardt 修正

**图 3-3 加权和模型的 Levenberg-Marquardt 修正攻击结果**

| 原图 | 攻击图 | 扰动图 |
|---|---|---|
| `datasets/1.png`<br>原图：sports car<br>14.517% | `outputs/final_square_validation/weighted_newton_lm/manual/1/attacked.png`<br>攻击图：pickup<br>26.3531 | `outputs/final_square_validation/weighted_newton_lm/manual/1/perturbation.png`<br>扰动图<br>L2=7.1967 |
| `datasets/2.png`<br>原图：Persian cat<br>9.729% | `outputs/final_square_validation/weighted_newton_lm/manual/2/attacked.png`<br>攻击图：paper towel<br>47.7373 | `outputs/final_square_validation/weighted_newton_lm/manual/2/perturbation.png`<br>扰动图<br>L2=11.6113 |
| `datasets/3.png`<br>原图：Windsor tie<br>12.536% | `outputs/final_square_validation/weighted_newton_lm/manual/3/attacked.png`<br>攻击图：hog<br>48.5944 | `outputs/final_square_validation/weighted_newton_lm/manual/3/perturbation.png`<br>扰动图<br>L2=9.5552 |
| `datasets/4.png`<br>原图：daisy<br>13.113% | `outputs/final_square_validation/weighted_newton_lm/manual/4/attacked.png`<br>攻击图：handkerchief<br>39.9955 | `outputs/final_square_validation/weighted_newton_lm/manual/4/perturbation.png`<br>扰动图<br>L2=9.1524 |

### 3.3.4 加权和模型：DFP 变尺度法

**图 3-4 加权和模型的 DFP 变尺度攻击结果**

| 原图 | 攻击图 | 扰动图 |
|---|---|---|
| `datasets/1.png`<br>原图：sports car<br>14.517% | `outputs/final_square_validation/weighted_dfp/manual/1/attacked.png`<br>攻击图：grille<br>25.3732 | `outputs/final_square_validation/weighted_dfp/manual/1/perturbation.png`<br>扰动图<br>L2=7.7795 |
| `datasets/2.png`<br>原图：Persian cat<br>9.729% | `outputs/final_square_validation/weighted_dfp/manual/2/attacked.png`<br>攻击图：paper towel<br>51.5791 | `outputs/final_square_validation/weighted_dfp/manual/2/perturbation.png`<br>扰动图<br>L2=18.5043 |
| `datasets/3.png`<br>原图：Windsor tie<br>12.536% | `outputs/final_square_validation/weighted_dfp/manual/3/attacked.png`<br>攻击图：hog<br>27.3859 | `outputs/final_square_validation/weighted_dfp/manual/3/perturbation.png`<br>扰动图<br>L2=7.8149 |
| `datasets/4.png`<br>原图：daisy<br>13.113% | `outputs/final_square_validation/weighted_dfp/manual/4/attacked.png`<br>攻击图：handkerchief<br>41.1281 | `outputs/final_square_validation/weighted_dfp/manual/4/perturbation.png`<br>扰动图<br>L2=13.4634 |

### 3.3.5 加权和模型：Fletcher-Reeves 共轭梯度法

**图 3-5 加权和模型的 Fletcher-Reeves 共轭梯度攻击结果**

| 原图 | 攻击图 | 扰动图 |
|---|---|---|
| `datasets/1.png`<br>原图：sports car<br>14.517% | `outputs/final_square_validation/weighted_conjugate_fr/manual/1/attacked.png`<br>攻击图：grille<br>28.5023 | `outputs/final_square_validation/weighted_conjugate_fr/manual/1/perturbation.png`<br>扰动图<br>L2=11.6288 |
| `datasets/2.png`<br>原图：Persian cat<br>9.729% | `outputs/final_square_validation/weighted_conjugate_fr/manual/2/attacked.png`<br>攻击图：paper towel<br>51.9728 | `outputs/final_square_validation/weighted_conjugate_fr/manual/2/perturbation.png`<br>扰动图<br>L2=16.4981 |
| `datasets/3.png`<br>原图：Windsor tie<br>12.536% | `outputs/final_square_validation/weighted_conjugate_fr/manual/3/attacked.png`<br>攻击图：hog<br>27.6661 | `outputs/final_square_validation/weighted_conjugate_fr/manual/3/perturbation.png`<br>扰动图<br>L2=11.1521 |
| `datasets/4.png`<br>原图：daisy<br>13.113% | `outputs/final_square_validation/weighted_conjugate_fr/manual/4/attacked.png`<br>攻击图：handkerchief<br>41.1274 | `outputs/final_square_validation/weighted_conjugate_fr/manual/4/perturbation.png`<br>扰动图<br>L2=12.6655 |

### 3.3.6 加权和模型：Polak-Ribiere+ 共轭梯度法

**图 3-6 加权和模型的 Polak-Ribiere+ 共轭梯度攻击结果**

| 原图 | 攻击图 | 扰动图 |
|---|---|---|
| `datasets/1.png`<br>原图：sports car<br>14.517% | `outputs/final_square_validation/weighted_conjugate_pr_plus/manual/1/attacked.png`<br>攻击图：grille<br>28.3311 | `outputs/final_square_validation/weighted_conjugate_pr_plus/manual/1/perturbation.png`<br>扰动图<br>L2=10.1449 |
| `datasets/2.png`<br>原图：Persian cat<br>9.729% | `outputs/final_square_validation/weighted_conjugate_pr_plus/manual/2/attacked.png`<br>攻击图：paper towel<br>53.9945 | `outputs/final_square_validation/weighted_conjugate_pr_plus/manual/2/perturbation.png`<br>扰动图<br>L2=16.4456 |
| `datasets/3.png`<br>原图：Windsor tie<br>12.536% | `outputs/final_square_validation/weighted_conjugate_pr_plus/manual/3/attacked.png`<br>攻击图：hog<br>35.4590 | `outputs/final_square_validation/weighted_conjugate_pr_plus/manual/3/perturbation.png`<br>扰动图<br>L2=14.1171 |
| `datasets/4.png`<br>原图：daisy<br>13.113% | `outputs/final_square_validation/weighted_conjugate_pr_plus/manual/4/attacked.png`<br>攻击图：handkerchief<br>41.0890 | `outputs/final_square_validation/weighted_conjugate_pr_plus/manual/4/perturbation.png`<br>扰动图<br>L2=13.7842 |

### 3.3.7 Courant 外点罚函数法

**图 3-7 Courant 外点罚函数攻击结果**

| 原图 | 攻击图 | 扰动图 |
|---|---|---|
| `datasets/1.png`<br>原图：sports car<br>14.517% | `outputs/final_square_validation/external_point/manual/1/attacked.png`<br>攻击图：convertible<br>7.2773 | `outputs/final_square_validation/external_point/manual/1/perturbation.png`<br>扰动图<br>L2=7.0175 |
| `datasets/2.png`<br>原图：Persian cat<br>9.729% | `outputs/final_square_validation/external_point/manual/2/attacked.png`<br>攻击图：paper towel<br>13.7810 | `outputs/final_square_validation/external_point/manual/2/perturbation.png`<br>扰动图<br>L2=6.8265 |
| `datasets/3.png`<br>原图：Windsor tie<br>12.536% | `outputs/final_square_validation/external_point/manual/3/attacked.png`<br>攻击图：groom<br>7.0968 | `outputs/final_square_validation/external_point/manual/3/perturbation.png`<br>扰动图<br>L2=6.0085 |
| `datasets/4.png`<br>原图：daisy<br>13.113% | `outputs/final_square_validation/external_point/manual/4/attacked.png`<br>攻击图：handkerchief<br>10.5680 | `outputs/final_square_validation/external_point/manual/4/perturbation.png`<br>扰动图<br>L2=6.7681 |

### 3.3.8 Softplus 光滑外点罚函数法

**图 3-8 Softplus 光滑外点罚函数攻击结果**

| 原图 | 攻击图 | 扰动图 |
|---|---|---|
| `datasets/1.png`<br>原图：sports car<br>14.517% | `outputs/final_square_validation/external_point_softplus/manual/1/attacked.png`<br>攻击图：convertible<br>7.2773 | `outputs/final_square_validation/external_point_softplus/manual/1/perturbation.png`<br>扰动图<br>L2=7.0175 |
| `datasets/2.png`<br>原图：Persian cat<br>9.729% | `outputs/final_square_validation/external_point_softplus/manual/2/attacked.png`<br>攻击图：paper towel<br>13.7810 | `outputs/final_square_validation/external_point_softplus/manual/2/perturbation.png`<br>扰动图<br>L2=6.8265 |
| `datasets/3.png`<br>原图：Windsor tie<br>12.536% | `outputs/final_square_validation/external_point_softplus/manual/3/attacked.png`<br>攻击图：groom<br>7.0968 | `outputs/final_square_validation/external_point_softplus/manual/3/perturbation.png`<br>扰动图<br>L2=6.0085 |
| `datasets/4.png`<br>原图：daisy<br>13.113% | `outputs/final_square_validation/external_point_softplus/manual/4/attacked.png`<br>攻击图：handkerchief<br>10.5680 | `outputs/final_square_validation/external_point_softplus/manual/4/perturbation.png`<br>扰动图<br>L2=6.7681 |

### 3.3.9 二范数球投影梯度法

**图 3-9 二范数球投影梯度攻击结果**

| 原图 | 攻击图 | 扰动图 |
|---|---|---|
| `datasets/1.png`<br>原图：sports car<br>14.517% | `outputs/final_square_validation/projected_gradient/manual/1/attacked.png`<br>攻击图：convertible<br>13.5178 | `outputs/final_square_validation/projected_gradient/manual/1/perturbation.png`<br>扰动图<br>L2=1.2281 |
| `datasets/2.png`<br>原图：Persian cat<br>9.729% | `outputs/final_square_validation/projected_gradient/manual/2/attacked.png`<br>攻击图：paper towel<br>38.9521 | `outputs/final_square_validation/projected_gradient/manual/2/perturbation.png`<br>扰动图<br>L2=3.5736 |
| `datasets/3.png`<br>原图：Windsor tie<br>12.536% | `outputs/final_square_validation/projected_gradient/manual/3/attacked.png`<br>攻击图：military uniform<br>44.0568 | `outputs/final_square_validation/projected_gradient/manual/3/perturbation.png`<br>扰动图<br>L2=3.8612 |
| `datasets/4.png`<br>原图：daisy<br>13.113% | `outputs/final_square_validation/projected_gradient/manual/4/attacked.png`<br>攻击图：handkerchief<br>29.3330 | `outputs/final_square_validation/projected_gradient/manual/4/perturbation.png`<br>扰动图<br>L2=1.8532 |

### 3.3.10 对比与分析

| 方法 | 后端 | 成功率 | 平均耗时/s | 最大耗时/s | 平均决策函数值 | 平均 L2 |
|---|---|---:|---:|---:|---:|---:|
| 加权和-最速下降 | manual | 4/4 | 0.238 | 0.244 | 39.138 | 9.728 |
| 加权和-最速下降 | efficient | 4/4 | 0.499 | 0.511 | 82.402 | 5.843 |
| 加权和-牛顿法 | manual | 4/4 | 6.901 | 10.782 | 37.421 | 9.732 |
| 加权和-牛顿法 | efficient | 4/4 | 0.491 | 0.508 | 82.402 | 5.843 |
| 加权和-LM 修正 | manual | 4/4 | 6.906 | 10.502 | 40.670 | 9.379 |
| 加权和-LM 修正 | efficient | 4/4 | 0.502 | 0.515 | 82.402 | 5.843 |
| 加权和-DFP 变尺度 | manual | 4/4 | 0.232 | 0.244 | 36.367 | 11.891 |
| 加权和-DFP 变尺度 | efficient | 4/4 | 0.500 | 0.517 | 82.402 | 5.843 |
| 加权和-FR 共轭梯度 | manual | 4/4 | 0.271 | 0.301 | 37.317 | 12.986 |
| 加权和-FR 共轭梯度 | efficient | 4/4 | 0.494 | 0.510 | 82.402 | 5.843 |
| 加权和-PR+ 共轭梯度 | manual | 4/4 | 0.280 | 0.326 | 39.718 | 13.623 |
| 加权和-PR+ 共轭梯度 | efficient | 4/4 | 0.497 | 0.512 | 82.402 | 5.843 |
| Courant 外点罚函数 | manual | 4/4 | 0.234 | 0.250 | 9.681 | 6.655 |
| Courant 外点罚函数 | efficient | 4/4 | 0.831 | 1.362 | 5.004 | 0.255 |
| Softplus 光滑外点罚函数 | manual | 4/4 | 0.247 | 0.271 | 9.681 | 6.655 |
| Softplus 光滑外点罚函数 | efficient | 4/4 | 1.344 | 1.428 | 5.492 | 0.201 |
| 二范数球投影梯度 | manual | 4/4 | 0.229 | 0.234 | 31.465 | 2.629 |
| 二范数球投影梯度 | efficient | 4/4 | 0.218 | 0.222 | 31.465 | 2.629 |

第 3 章的数值法都取得 100% 成功率，但不同模型对应的运筹学含义并不相同。加权和模型把“攻击成功”和“扰动较小”写成同一个无约束目标，它更接近目标规划中的加权偏差处理：权重给定后，求解器追求的是综合指标的下降，而不是严格把扰动压在某个硬边界上。因此该类方法的决策函数值普遍较大，说明攻击约束存在较大余量；同时 L2 也明显大于第 2 章的二范数约束解，体现出软约束模型容易产生“过度可行”的解。

从下降迭代法角度看，manual 后端的最速下降、DFP 和共轭梯度均在 0.3 秒内完成，符合课件中“梯度型方法计算量小、适合大规模问题”的特点。牛顿法和 LM 修正平均耗时约 6.9 秒，因为它们利用二阶信息或二阶近似，单步方向质量更高，但需要额外计算曲率信息。LM 修正的平均 L2 为 9.379，略低于普通牛顿法的 9.732，说明把非正定或病态 Hessian 修正为更稳定的下降方向后，局部求解更平稳。

外点罚函数和投影梯度体现了两类不同的约束处理。外点法通过罚因子把约束违反量并入目标函数，manual 后端给出的平均 L2 为 6.655，攻击余量较充足；efficient 后端则把平均 L2 压到 0.201 到 0.255，说明其更接近“刚好可行”的边界解。投影梯度法每步都把扰动映射回二范数球可行域，平均 L2 为 2.629，兼顾了较强攻击效果和明确的原始可行性。若从课程中的可行方向法看，投影梯度的优势在于每一步都保持可行候选，而外点法的优势在于可以先在可行域外搜索，再通过罚函数逐步逼近约束边界。

## 4.5 实验结果

### 4.5.1 对偶损失双层迭代

**图 4-1 对偶损失双层迭代攻击结果**

| 原图 | 攻击图 | 扰动图 |
|---|---|---|
| `datasets/1.png`<br>原图：sports car<br>14.517% | `outputs/final_square_validation/dual_loss/manual/1/attacked.png`<br>攻击图：convertible<br>7.6259 | `outputs/final_square_validation/dual_loss/manual/1/perturbation.png`<br>扰动图<br>L2=0.6092 |
| `datasets/2.png`<br>原图：Persian cat<br>9.729% | `outputs/final_square_validation/dual_loss/manual/2/attacked.png`<br>攻击图：paper towel<br>34.4527 | `outputs/final_square_validation/dual_loss/manual/2/perturbation.png`<br>扰动图<br>L2=2.1512 |
| `datasets/3.png`<br>原图：Windsor tie<br>12.536% | `outputs/final_square_validation/dual_loss/manual/3/attacked.png`<br>攻击图：military uniform<br>33.5000 | `outputs/final_square_validation/dual_loss/manual/3/perturbation.png`<br>扰动图<br>L2=1.9975 |
| `datasets/4.png`<br>原图：daisy<br>13.113% | `outputs/final_square_validation/dual_loss/manual/4/attacked.png`<br>攻击图：handkerchief<br>18.8496 | `outputs/final_square_validation/dual_loss/manual/4/perturbation.png`<br>扰动图<br>L2=0.9042 |

### 4.5.2 对比与分析

| 方法 | 后端 | 成功率 | 平均耗时/s | 最大耗时/s | 平均决策函数值 | 平均 L2 |
|---|---|---:|---:|---:|---:|---:|
| 对偶损失双层迭代 | manual | 4/4 | 0.990 | 1.017 | 23.607 | 1.416 |
| 对偶损失双层迭代 | efficient | 4/4 | 1.990 | 1.999 | 64.246 | 2.759 |

对偶方法同样在四张图片上全部成功。manual 后端平均 L2 为 1.416，是连续攻击实验中扰动较小的一组；efficient 后端平均决策函数值更高，说明其选择了攻击余量更大的可行点，但代价是平均 L2 上升到 2.759。这个现象与对偶乘子的解释一致：乘子可以看作攻击阈值约束的边际权重，权重更新越强调满足攻击约束，越容易得到决策函数值更大的解。

需要注意的是，本问题不是线性规划或凸规划，弱对偶、强对偶以及互补松弛不能直接提供全局最优证书。这里的对偶求解更适合作为一种原问题-对偶问题交替调整的数值策略：内层寻找当前乘子下的原始可行候选，外层依据约束余量修正乘子。实验中所有样本都达到攻击成功，说明该双层策略能稳定找到可行点；manual 后端较小的 L2 表明其更偏向“接近边界的满意解”，efficient 后端较高的决策函数值则表明其更偏向“安全余量更大的可行解”。

## 5.3 实验结果

### 5.3.1 定向识别为 toilet tissue

**图 5-1 定向 toilet tissue 攻击结果**

| 原图 | 攻击图 | 扰动图 |
|---|---|---|
| `datasets/1.png`<br>原图：sports car<br>14.517% | `outputs/final_square_validation/toilet_tissue/manual/1/attacked.png`<br>攻击图：toilet tissue<br>3.8849 | `outputs/final_square_validation/toilet_tissue/manual/1/perturbation.png`<br>扰动图<br>L2=1.4600 |
| `datasets/2.png`<br>原图：Persian cat<br>9.729% | `outputs/final_square_validation/toilet_tissue/manual/2/attacked.png`<br>攻击图：toilet tissue<br>5.1411 | `outputs/final_square_validation/toilet_tissue/manual/2/perturbation.png`<br>扰动图<br>L2=0.5970 |
| `datasets/3.png`<br>原图：Windsor tie<br>12.536% | `outputs/final_square_validation/toilet_tissue/manual/3/attacked.png`<br>攻击图：toilet tissue<br>3.4324 | `outputs/final_square_validation/toilet_tissue/manual/3/perturbation.png`<br>扰动图<br>L2=0.9810 |
| `datasets/4.png`<br>原图：daisy<br>13.113% | `outputs/final_square_validation/toilet_tissue/manual/4/attacked.png`<br>攻击图：toilet tissue<br>0.1079 | `outputs/final_square_validation/toilet_tissue/manual/4/perturbation.png`<br>扰动图<br>L2=1.6577 |

### 5.3.2 对比与分析

| 方法 | 后端 | 成功率 | 平均耗时/s | 最大耗时/s | 平均决策函数值 | 平均 L2 |
|---|---|---:|---:|---:|---:|---:|
| 定向 toilet tissue 攻击 | manual | 4/4 | 1.120 | 1.155 | 3.142 | 1.174 |
| 定向 toilet tissue 攻击 | efficient | 4/4 | 2.128 | 2.180 | 7.670 | 1.407 |

定向攻击要求模型不仅离开原类别，还必须进入指定类别 `toilet tissue`，因此比非定向攻击多了一层目标约束。若按目标规划语言理解，目标值是“toilet tissue 类别相对原类别及其他类别取得正间隔”，决策函数值可以视为目标达成后的正偏差或安全余量。四张图均成功，说明该定向间隔约束在给定扰动预算下可行。

从样本差异看，波斯猫只需 L2=0.5970 即可达到 5.1411 的目标间隔，说明其在模型特征空间中较容易被定向推向 `toilet tissue`；雏菊的 L2=1.6577 最大，但决策函数值仅 0.1079，几乎贴近阈值 0.1，是典型的“刚好可行”解。这一对比有课程意义：同样满足原始可行性时，约束余量大小揭示了样本到目标决策区域的距离差异。efficient 后端的平均决策函数值更大，表明其更重视目标约束余量；manual 后端平均 L2 更小，表明其更重视扰动成本。

## 6.3 实验结果

### 6.3.1 逆向贪心剪枝

**图 6-1 逆向贪心剪枝文字攻击结果**

| 原图 | 攻击图 | 扰动图 |
|---|---|---|
| `datasets/1.png`<br>原图：sports car<br>14.517% | `outputs/final_square_validation/text_reverse_greedy/manual/1/attacked.png`<br>攻击图：convertible<br>0.1450 | `outputs/final_square_validation/text_reverse_greedy/manual/1/perturbation.png`<br>扰动图<br>面积=37；L2=6.2661 |
| `datasets/2.png`<br>原图：Persian cat<br>9.729% | `outputs/final_square_validation/text_reverse_greedy/manual/2/attacked.png`<br>攻击图：Angora<br>0.1115 | `outputs/final_square_validation/text_reverse_greedy/manual/2/perturbation.png`<br>扰动图<br>面积=57；L2=10.7757 |
| `datasets/3.png`<br>原图：Windsor tie<br>12.536% | `outputs/final_square_validation/text_reverse_greedy/manual/3/attacked.png`<br>攻击图：oboe<br>0.1153 | `outputs/final_square_validation/text_reverse_greedy/manual/3/perturbation.png`<br>扰动图<br>面积=145；L2=9.9115 |
| `datasets/4.png`<br>原图：daisy<br>13.113% | `outputs/final_square_validation/text_reverse_greedy/manual/4/attacked.png`<br>攻击图：handkerchief<br>0.1895 | `outputs/final_square_validation/text_reverse_greedy/manual/4/perturbation.png`<br>扰动图<br>面积=34；L2=7.4769 |

### 6.3.2 一阶线性化 MILP

**图 6-2 一阶线性化 MILP 文字攻击结果**

| 原图 | 攻击图 | 扰动图 |
|---|---|---|
| `datasets/1.png`<br>原图：sports car<br>14.517% | `outputs/final_square_validation/text_linearized_milp/manual/1/attacked.png`<br>攻击图：convertible<br>0.1045 | `outputs/final_square_validation/text_linearized_milp/manual/1/perturbation.png`<br>扰动图<br>面积=40；L2=6.4599 |
| `datasets/2.png`<br>原图：Persian cat<br>9.729% | `outputs/final_square_validation/text_linearized_milp/manual/2/attacked.png`<br>攻击图：Angora<br>0.1532 | `outputs/final_square_validation/text_linearized_milp/manual/2/perturbation.png`<br>扰动图<br>面积=56；L2=10.5717 |
| `datasets/3.png`<br>原图：Windsor tie<br>12.536% | `outputs/final_square_validation/text_linearized_milp/manual/3/attacked.png`<br>攻击图：oboe<br>1.1622 | `outputs/final_square_validation/text_linearized_milp/manual/3/perturbation.png`<br>扰动图<br>面积=323；L2=14.4669 |
| `datasets/4.png`<br>原图：daisy<br>13.113% | `outputs/final_square_validation/text_linearized_milp/manual/4/attacked.png`<br>攻击图：handkerchief<br>0.1895 | `outputs/final_square_validation/text_linearized_milp/manual/4/perturbation.png`<br>扰动图<br>面积=34；L2=7.4769 |

### 6.3.3 对比与分析

| 方法 | 后端 | 成功率 | 平均耗时/s | 最大耗时/s | 平均决策函数值 | 平均 L2 | 平均面积 |
|---|---|---:|---:|---:|---:|---:|---:|
| 逆向贪心剪枝 | manual | 4/4 | 23.549 | 44.832 | 0.140 | 8.608 | 68.3 |
| 逆向贪心剪枝 | efficient | 4/4 | 23.495 | 45.210 | 0.140 | 8.608 | 68.3 |
| 一阶线性化 MILP | manual | 4/4 | 12.056 | 21.629 | 0.402 | 9.744 | 113.3 |
| 一阶线性化 MILP | efficient | 4/4 | 14.583 | 30.528 | 0.139 | 10.182 | 121.5 |

文字攻击与前几章的连续扰动不同，本章的决策变量是是否保留某个黑色文字像素，属于 0-1 非线性规划。若把候选文字像素看作格点图的顶点，则“张天羽”每个笔画的保留区域必须是连通子图；若再引入辅助有向弧和流量变量，连通性可以转化为类似最大流模型中的容量约束与流量平衡约束。这使第 6 章更接近组合优化和网络优化问题，而不只是连续非线性规划问题。

逆向贪心剪枝在四张图片上都得到更小的平均面积，尤其在跑车和雏菊上分别只需 37 和 34 个黑色像素即可成功。其机制相当于从一个可行解出发，不断删除当前边际贡献较低的像素，并在每一步检查攻击约束、笔画保留率和连通性。该方法的优点是最终面积小、结果直观；缺点是每次删除都需要重新验证真实模型和图结构约束，因此温莎结样本耗时达到 44.832 秒。

一阶线性化 MILP 把神经网络攻击约束在当前点附近近似为线性不等式，并与 0-1 面积目标、模板约束和网络流连通约束共同求解。它的平均耗时低于逆向贪心，但平均面积更大，尤其温莎结样本面积达到 323。这说明线性化模型在局部近似意义下能较快找到满足约束的整数可行解，却可能因为安全余量或局部梯度估计而保留更多像素。就课程术语而言，逆向贪心强调从可行解出发的局部改进，MILP 强调把 0-1 变量、线性化攻击收益和网络流约束纳入统一标准形式。两者都不是原始非凸 0-1 模型的全局最优证书，但都给出了经过真实模型验证的可行满意解。
