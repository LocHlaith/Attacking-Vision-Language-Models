# 攻击视觉大模型：运筹学算法实现

本仓库严格按照 `Codex必读.md`、`papers/draft.txt` 与 `slides` 课件实现。代码将图像攻击写成非线性规划、对偶问题、网络流模型和混合整数线性规划，并同时提供手搓后端与高效实验后端。

## 环境

项目已创建 `.venv`。当前环境使用支持 RTX 5070 的 PyTorch CUDA 版本。

```powershell
.venv\Scripts\activate
python -m pytest -q
```

若需要重建环境：

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu128 torch torchvision
uv pip install --python .venv\Scripts\python.exe numpy pandas pillow scipy pytest
```

## 已实现算法

- 解析法：一阶 Lagrange 满意解、二阶 KKT 近似、Fritz John 与 KKT 残差诊断。
- 数值法：最速下降法、牛顿法、Levenberg-Marquardt 修正、DFP 变尺度法、Fletcher-Reeves 与 Polak-Ribiere 自动重置共轭梯度法。
- 约束数值法：原始 Courant 外点罚函数、Softplus 光滑近似、二范数球投影梯度法。
- 对偶方法：攻击阈值模型的近似原-对偶双层迭代。
- 定向模型：将输入识别为第 1000 类 `toilet tissue`。
- 文字模型：0-1 非线性模型、手搓增广链最大流连通检验、网络流混合整数线性规划、逐次线性近似、反向贪心剪枝。

通用手搓求解器位于 `avlm_or/solvers/manual.py` 和 `avlm_or/solvers/graph.py`。高效实验后端位于 `avlm_or/solvers/efficient.py`，文字 MILP 使用 SciPy 高效求解器。

## 运行实验

四张原图只记录一次：

```powershell
python -m avlm_or.run baseline
```

若实验需要使用人工真实类别而不是模型原始 Top-1 决策，可提供参考类别文件：

```powershell
python -m avlm_or.run baseline --reference-labels datasets/reference_labels.json
python -m avlm_or.run all-continuous --backend manual --reference-labels datasets/reference_labels.json
```

参考类别文件只覆盖 `original_class`，模型的实际 `decision_class` 仍会如实记录。当前
`datasets/reference_labels.json` 为空，因为四张实验图片均直接使用模型的 Top-1 决策。

运行一种连续攻击：

```powershell
python -m avlm_or.run attack --algorithm analytic_first_order --backend manual
python -m avlm_or.run attack --algorithm weighted_dfp --backend manual --max-iterations 30
python -m avlm_or.run attack --algorithm toilet_tissue --backend efficient
python -m avlm_or.run attack --algorithm dual_loss --restarts 3
```

运行全部连续模型：

```powershell
python -m avlm_or.run all-continuous --backend manual
```

运行文字攻击：

```powershell
python -m avlm_or.run text --text-method reverse_greedy --backend manual --max-checks 200
python -m avlm_or.run text --text-method linearized_milp --backend efficient
python -m avlm_or.run text --text-method linearized_milp --backend efficient --template-limit 1 --milp-time-limit 15 --linearization-iterations 1
python -m avlm_or.run text --text-method reverse_greedy --backend efficient --template-limit 20 --text-font-sizes 20,24,28,32,36,40 --text-angles=-90,-60,-45,-30,0,30,45,60,90
```

文字模板默认使用“张天羽”三字，搜索 `10` 至 `64` 号字体、顶部/中部/底部位置，
以及 `-90` 至 `90` 度旋转角度。候选模板优先按完整置黑可行性和面积排序。

## 输出

结果统一写入 `outputs/<algorithm>/<backend>/`：

- `results.csv`：英文实验记录。
- `<image>/attacked.png`：攻击图。
- `<image>/perturbation.png`：扰动图。
- `<image>/mask.png`：文字攻击的 0-1 掩码。

CSV 包含模型决策、决策函数值与阈值、扰动值与阈值、文字扰动面积、成功状态、耗时、迭代次数和终止信息。

## 说明

- 图片没有附带人工真实标签，因此代码按论文实验语境，将模型对原图的决策作为被攻击的原类别。
- 手搓 DFP 使用秩二校正的精确矩阵作用表示，避免显式存储 `150528 x 150528` 矩阵，但更新公式与课件 DFP 公式一致。
- 二阶 KKT 近似使用手搓共轭方向线性方程求解器计算矩阵作用，不显式形成完整 Hessian。
- `manual` 后端用于课程提交；`efficient` 后端用于实际实验，不应放入课程提交附件。
