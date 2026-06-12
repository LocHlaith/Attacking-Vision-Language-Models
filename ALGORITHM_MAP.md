# 论文公式与代码对应表

本表用于说明各实现均直接对应 `papers/draft.txt` 中的模型与算法。

| 论文内容 | 公式 | 代码 |
|---|---:|---|
| 一阶 Lagrange 满意解 | (2-37)、(2-51) | `avlm_or.attacks.analytic_first_order` |
| 二阶 KKT 近似方程 | (2-52) 至 (2-58) | `avlm_or.attacks.second_order_ball_approximation` |
| Fritz John、KKT 条件检验 | (2-38) 至 (2-49)、(4-13) 至 (4-16) | `avlm_or.optimality` |
| 加权和无约束模型 | (3-1) 至 (3-3) | `avlm_or.objectives.weighted_sum` |
| 最速下降法 | (3-18) 至 (3-20) | `avlm_or.solvers.manual.steepest_descent` |
| 牛顿法与 LM 修正 | (3-21) 至 (3-28) | `avlm_or.solvers.manual.levenberg_marquardt_newton` |
| DFP 变尺度法 | (3-29) 至 (3-32) | `avlm_or.solvers.manual.dfp_change_scale` |
| Fletcher-Reeves、PR+ 自动重置 | (3-35) 至 (3-38) | `avlm_or.solvers.manual.nonlinear_conjugate_gradient` |
| Courant 外点罚函数 | (3-5) 至 (3-8) | `avlm_or.solvers.manual.external_point_method` |
| Softplus 光滑近似罚函数 | (3-9)、(3-10) | `avlm_or.attacks.smooth_external_point_attack` |
| 二范数约束投影梯度与映射 | (3-39) 至 (3-43) | `avlm_or.solvers.manual.projected_gradient` |
| 近似原-对偶双层迭代 | (4-17) 至 (4-23) | `avlm_or.attacks.approximate_primal_dual` |
| 厕纸定向分类间隔与双层迭代 | (5-2) 至 (5-14) | `avlm_or.objectives.target_margin`、`avlm_or.attacks.toilet_tissue_attack` |
| 0-1 非线性文字模型 | (6-8) 至 (6-16) | `avlm_or.text_attack.black_perturbation`、`true_margin` |
| 网络流连通性模型 | (6-17)、(6-18) | `avlm_or.solvers.graph`、`efficient_linearized_milp` |
| 一阶近似混合整数线性规划 | (6-19) 至 (6-25) | `avlm_or.text_attack.linearized_gain`、`efficient_linearized_milp` |
| 逐次线性化与反向贪心剪枝 | 第 6.2 节 | `linearized_then_prune`、`reverse_greedy_pruning` |

`avlm_or/solvers/manual.py` 与 `avlm_or/solvers/graph.py` 为课程提交使用的手搓求解器；`avlm_or/solvers/efficient.py` 以及文字 MILP 的高效求解函数只用于实际实验。
