# 第三部分：实验框架、调参与批量运行

负责人：CHEN Yuchan（@Sans）。本模块把 proposal 第 4–5 节的实验设计变成可以复现、可以中断续跑的代码：问题与配对初始点的生成、pilot、两种调参方式、批量评估和失败日志，并把整洁的结果表交给第 4 部分做统计。英文报告段落见 [part3_report.tex](part3_report.tex)。

**边界**：本模块不实现、不修改任何优化器，所有运行都通过公共的 `run(problem, initial, RunConfig)`；也不做 bootstrap、成功率区间等统计推断和结果图（第 4 部分负责）。smoke 和 pilot 的数字都不是研究结论。

## 1. 文件一览

| 文件 | 内容 |
| --- | --- |
| `src/coordinate_geometry/part3_design.py` | 谱、旋转、配对初始点、seed key、各阶段的运行单元 |
| `src/coordinate_geometry/part3_tuning.py` | 候选网格、稳定区间上限、评分、选择规则、pilot 范围调整 |
| `src/coordinate_geometry/part3_experiments.py` | pilot / tune / evaluate / report 四个阶段、并行与续跑、输出表 |
| `src/coordinate_geometry/verify_part3.py` | 用另一套代码从 seed 重建实例，复跑抽样 run，重算全部选择 |
| `configs/part3.json` | 主实验协议 |
| `configs/part3_smoke.json` | 几秒钟的端到端检查，只验证流程和格式 |
| `scripts/part3.ps1` | Windows 一键脚本（测试 → 运行 → 独立复核） |
| `tests/test_part3.py` | 生成器、seed 隔离、选择规则、pilot 规则、端到端与续跑测试 |

`problems.py` 与 `runner.py` 中已有的公共接口保持原样；新代码只在旁边增加文件。

## 2. 实例生成

**谱**。固定 $L=1$，$\mu=1/\kappa$：

```math
D=\operatorname{diag}(d_1,\dots,d_p),\qquad d_i=\kappa^{-(p-i)/(p-1)} .
```

二维时就是 $D=\operatorname{diag}(1/\kappa,1)$；$p=20$ 时对数等距；$\kappa=1$ 时 $D=I$，作为各向同性对照。$\kappa$ 取 $\{1,10,10^2,10^3,10^4\}$。

**旋转**。二维评估使用精确角度 $Q_\theta$，$\theta\in\{0^\circ,15^\circ,\dots,90^\circ\}$；$0^\circ$ 是对角参考，$90^\circ$ 是带符号置换对照。高维使用公共的 `random_orthogonal`（高斯 QR 加对角符号修正，即 Haar 分布）。

**配对初始点**（proposal 式 (3)）。从高斯方向 $u$ 出发

```math
z_0=\sqrt{\frac{2}{u^\top D u}}\,u,\qquad w_0=Qz_0,\qquad f_I(z_0)=f_Q(w_0)=1,\quad \lVert w_0\rVert=\lVert z_0\rVert .
```

这与公共 `matched_pair` 对同一个 $u$ 给出的 $z_0$ 逐位相同（有测试）。配置中 `initial_normalization: "unit_norm"` 和 `initial_scale` 用于 proposal 提到的单位范数与尺度检查。

**seed 流**。每个随机对象都有自己的 `SeedSequence(root_seed, spawn_key)`，spawn key 为

```text
(阶段, 用途, 研究, κ, 调参方式, pilot 轮次, 方向编号, 序号)
阶段: pilot=1, tuning=2, evaluation=3；用途: 旋转=1, 初始方向=2
```

所以 pilot、调参、评估的实例互不重叠；任何一次 run 都能只凭记录里的 key 重建，不需要按顺序重放整个实验。所有 seed key 和旋转矩阵来源都写进每条记录。

## 3. 研究设计

| 研究 | 评估单元 | 配对 | 重抽样单元 `cluster_id` |
| --- | --- | --- | --- |
| `planar`（$p=2$） | 每个 $\kappa$：7 个角度 × 10 个初始方向；方向在各角度间共享 | $(\theta,i)$ 与 $(0^\circ,i)$ 配对；$0^\circ$ 的参考 run 只跑一次，不重复计为独立观测 | 初始方向 $i$（跨角度一起重抽） |
| `p20`（$p=20$） | 每个 $\kappa$：10 个 Haar 旋转 × 每个 5 个初始方向 | $(Q_r,z_{r,i})$ 与 $(I,z_{r,i})$ 配对 | 旋转 $r$ |

算法：`gd`、`heavy_ball`（$\beta=0.9$）、`adam`、`eigenbasis_adam`（oracle 基）为主比较，`diagonal_gd` 为次要比较。每个评估单元都在三种方式下运行：

- **transfer**：每个 (算法, 研究, $\kappa$) 一个学习率，在 4 个调参方向（含对角基）× 5 个调参初始点上选出，再固定用于全部评估方向；
- **retune**：每个评估方向单独调一个学习率，使用 5 个与评估初始点不同的调参初始点；
- **theory**：GD 用 $\eta^*=2/(L+\mu)$，heavy ball 用 $(\eta^*_{HB},\beta^*_{HB})$（proposal 式 (7)(10)），只作使用谱信息的参考，不参加等预算比较。与第 1 部分 `optimal_*_parameters` 的数值一致（有测试）。

二维 transfer 的另外 3 个调参方向是从调参 seed 流抽取的 $(0^\circ,90^\circ)$ 均匀角度，因此一般不落在评估网格上；高维用调参 seed 流的 Haar 旋转。retune 在评估方向本身上调参（这是 retune 的定义），只是初始点来自调参流。

## 4. 调参规则

每个算法 16 个对数等距候选，每个候选、每个调参初始点跑满 5,000 次梯度调用（达到阈值也不提前停）。候选得分：

```math
\operatorname{score}(\eta)=\operatorname{median}\Big[\log_{10}\max\{r_B,10^{-16}\}\Big],\qquad r_B=\frac{f(w_B)}{f(w_0)},\ B=5000 .
```

- 任一调参 run 为 `nonfinite` 或 `explosive`（$r_t>10^{12}$），该候选记为最差分；只是预算内没到 $10^{-6}$ 不算失败。
- 平分取较小学习率（`tie_break: "smaller_rate"`，即 proposal 的规则）。
- 全部候选都失败时记为 tuning failure：不选学习率，评估表中对应行状态为 `not_run_tuning_failure`，仍保留在分母里。
- GD 与 heavy ball 的候选严格位于稳定区间内：上端取稳定上界的 0.999 倍，即 GD $\eta\le 1.998$、heavy ball $\eta\le 3.7962$（$\beta=0.9$ 时上界 $2(1+\beta)=3.8$）。

**评估**：50,000 次梯度调用，检查点 1,000 / 10,000 / 50,000；逐步记录首次达到 $r_t\le10^{-6}$ 的时间 $T$；`threshold_retained` 表示最终仍在阈值内；`max_recorded_relative_after_attainment` 给出达到后在记录点上的最大回升，用于检查越过阈值后又振荡的情况。

## 5. Pilot

pilot 在独立的 pilot seed 上，对每个 (算法, 研究, $\kappa$) 用 2 个方向（含对角基）× 3 个初始点、同样的 16 个候选和 5,000 次预算跑一遍，再用同一选择规则找最优。若最优在网格边界上：下端除以 10；上端乘以 10，但不越过稳定上界。调整后换新的 pilot seed 重跑，最多两次（共 3 轮）；仍未解决的记为 `unresolved_boundary`，作为范围限制报告。最后冻结全部候选，写入 `frozen_protocol.json`。

### 本次 pilot 结果（2026-10-18，`configs/part3.json`，2 个 worker，用时 201 秒）

共 5,760 次 pilot run（2,847 万次梯度调用，部分候选因爆炸提前结束），全部来自 pilot seed，之后不再使用。

| 算法 | 起始范围 | 轮次 | 冻结范围 | 边界情况 |
| --- | --- | --- | --- | --- |
| `gd` | $[0.002,\,1.998]$ | 1 | 不变 | 二维 $\kappa=10^3$ 和两项研究 $\kappa=10^4$ 的最优在上端，即稳定上界的 0.999 倍：标记为 `stability_capped`，不扩展 |
| `heavy_ball` | $[0.0038,\,3.7962]$ | 1 | 不变 | $\kappa=10^4$ 最优在稳定上限处（`stability_capped`）；$\kappa=1,10$ 时 16 个候选全部达到 $10^{-16}$，属于 `edge_plateau` |
| `adam` | $[10^{-4},\,1]$ | 2 | $[10^{-4},\,10]$ | 第 0 轮二维 $\kappa=10^4$ 的最优在上端且严格优于相邻候选，上端扩展 10 倍；第 1 轮（新 seed）所有最优都在网格内部 |
| `eigenbasis_adam` | $[10^{-4},\,1]$ | 1 | 不变 | 无 |
| `diagonal_gd` | $[0.002,\,2]$ | 1 | 不变 | 无 |

注意：Adam 与特征坐标 Adam 现在的网格不同（相邻候选比分别约为 2.15 和 1.85）。这是按"每个算法自己的范围"执行 proposal 规则的结果；比较这两者时应在报告中说明网格粒度不同。

### 实现中细化的两点（需组内确认）

1. **边界的判定**。$10^{-16}$ 的下限会让条件数小的格子里一整段候选同分（都达到下限）。按"平分取较小学习率"，这时会选中网格最下端，但最优集合其实延伸到网格内部，往下扩展范围永远找不到"边界最优"，只会让网格变粗。因此实现中规定：只有当边界候选**严格优于**相邻候选时才算边界最优；同分平台从边界开始的情况记为 `edge_plateau`，不扩展。这是对 proposal 规则的解释，不改变得分与选择本身。
2. **平分规则对达到时间的影响**。同样由于下限，"取较小学习率"会在平台中选出"仍能在 5,000 步内达到 $10^{-16}$ 的最慢学习率"。这对固定预算精度没有影响，但会让首次达到时间 $T$ 偏长。代码保留 proposal 的规则作为默认值，同时提供 `tuning.tie_break: "earliest_attainment"`（平分时先比中位达到时间，再取较小学习率）。改动它会改变协议哈希，需要在新目录重新 pilot。是否切换请组内决定，并在报告中写明。

## 6. 失败与截断

| `status` | 含义 | 处理 |
| --- | --- | --- |
| `finite_cap` | 跑满预算，没有数值失败或爆炸 | 是否达标看 `first_attainment` 与 `threshold_retained` |
| `explosive` | 某步 $r_t>10^{12}$ | 保留最后一个有限迭代点、触发步和轨迹 |
| `nonfinite` | 出现 inf / NaN（含 Adam 矩） | 同上 |
| `not_run_tuning_failure` | 该格子所有候选在调参时都失败 | 不运行，不填入人为的达到时间 |

所有非 `finite_cap` 记录（pilot、调参、评估）集中写入 `tables/failures.jsonl`，包含最后有限点、轨迹、配置和 seed key。

## 7. 输出（给第 4 部分）

`outputs/part3/tables/` 中：

| 表 | 一行是什么 | 关键字段 |
| --- | --- | --- |
| `evaluation_runs.csv` | 一次评估运行 | `regime`、`algorithm`、`study`、`kappa`、`orientation_id`、`angle_degrees`、`init_id`、`role`、`pair_id`、`reference_run_id`、`cluster_id`、`learning_rate`、`status`、`first_attainment`、`threshold_retained`、`r_1000`/`r_10000`/`r_50000`、seed key |
| `pairs.csv` | 一对（旋转 run, 对角参考 run） | `correspondence_expected`、`separation_step`（见 §9）、双方状态与达到时间、四类结果 `outcome`（both / diagonal_only / rotated_only / neither，与 `analysis.paired_attainment_counts` 一致）、仅双方都达到时的 `attainment_ratio`、各检查点的 $\Delta_B$（proposal 式 (32)，任一方在该检查点无有限值时留空） |
| `evaluation_traces.csv` | 每 1,000 步一个记录点 | `relative_loss`、该步更新前的梯度范数 |
| `failures.jsonl` | 一条非 `finite_cap` 记录 | 最后有限点、轨迹、配置 |
| `selected_rates.csv` / `tuning_scores.csv` | 一个调参格子 / 一个候选 | 选中的学习率、边界与平台标记、每个候选的得分与失败数 |
| `status_summary.csv` | 状态计数 | 便于先看失败分布 |

`pairs.csv` 只把数据配好，不计算区间；$\theta=0^\circ$ 的对角参考本身不再作为一对出现。`budget_summary.json` 记录各阶段梯度调用数和 5,000 / 50,000 两个时间范围的差别；`manifest.json` 记录 git commit、源文件哈希、Python/NumPy/BLAS、线程环境变量和每次调用的阶段耗时。

## 8. 运行与复核

```powershell
.\scripts\part3.ps1                         # 测试 + smoke
.\scripts\part3.ps1 -Stage all -Workers 6   # 主实验：pilot → tune → evaluate → report → 独立复核
```

```text
python -m coordinate_geometry.part3_experiments --config configs/part3.json --output outputs/part3 --stage all --workers 6 --plot
python -m coordinate_geometry.verify_part3 --output outputs/part3 --sample 60
```

- 每个阶段都可以中断后重新执行同一命令，已保存的 run 会跳过；同一 `run_id` 若规格不同会直接报错，避免两套协议混在一个目录。
- 冻结协议后修改 `tuning`、`pilot`、`algorithms`、阈值或 `root_seed` 会被拒绝；`studies` 可以按 pilot 的成本估计缩减。
- 并行 worker 使用 `spawn`，并设置 BLAS 单线程。多 worker 时每次 run 的计时彼此竞争 CPU，只能作参考，不能当作成本测量。
- `verify_part3` 不使用 `part3_design`：它从记录中的 seed key 重新构造谱、旋转和初始点，复跑抽样 run 并要求状态、达到时间和检查点逐位相同；独立重算全部调参选择；检查 seed 阶段隔离、配对共享同一 $z_0$、transfer 配对使用同一学习率、失败记录保留有限迭代点。

## 9. 本次主实验运行（2026-10-18/19，按上面冻结的协议）

这是按当前默认协议跑完的一版完整数据，供第 4 部分开发和组内检查。如果组内修改协议（例如 §5 的平分规则或研究范围），需要在新目录重新 pilot 和运行。

| 阶段 | 运行数 | 梯度调用 | 2 个 worker 的墙钟时间 |
| --- | --- | --- | --- |
| pilot | 5,760 | 2.85 × 10⁷ | 3.4 分钟 |
| 调参：transfer / retune | 52,000 | 0.78 × 10⁸ / 1.76 × 10⁸ | 57 分钟 |
| 评估：transfer / retune / theory | 10,200 | 2.13 × 10⁸ / 2.13 × 10⁸ / 0.85 × 10⁸ | 约 1 小时 51 分钟（中途按设计续跑一次） |

- 500 个调参格子全部选出学习率，没有 tuning failure；47 个最优在网格上端，全部是 GD / heavy ball 落在稳定上限处（`stability_capped`），没有范围限制；40 个 heavy ball 格子（$\kappa=1,10$）是 `edge_plateau`。
- 1,261 条非 `finite_cap` 记录全部来自次要方法 `diagonal_gd` 在 pilot 和调参中的大学习率候选（`explosive`），这些候选按规则记最差分。10,200 次评估运行全部是 `finite_cap`，所有配对双方都达到过 $10^{-6}$。
- `verify_part3`：每个阶段抽 150 次 run 逐位复跑一致，500 个选择独立重算一致，6,600 对配对检查通过。

### 给第 4、5 部分：Adam 在收敛后的尖峰与配对分离

`pairs.csv` 新增两列：`correspondence_expected`（精确算术下这一对应当逐步重合：GD、heavy ball、特征坐标 Adam 在同一设置下总是如此；Adam 与对角预条件只在带符号置换时如此）和 `separation_step`（两条记录轨迹的 $\log_{10} r$ 第一次相差超过 $10^{-6}$ 的记录步）。

本次数据中，所有 `correspondence_expected` 的配对在首次达到阈值之前都没有分离。GD、heavy ball 和对角预条件始终不分离（$|\Delta_B|\le 3\times10^{-10}$）。但 Adam 与特征坐标 Adam 的配对在之后的记录步上普遍分离（中位分离步为 17,000），$|\Delta_{10000}|$、$|\Delta_{50000}|$ 可达 8–13。原因已用单次复跑确认：损失降到极小后，梯度相对已存的 $\sqrt{v}$ 可以忽略，$v$ 按 $\beta_2^t$ 衰减，有效步长 $\eta/\sqrt{\hat v_t}$ 不断增大，直到超过 $2/\lambda$，迭代点从约 $10^{-250}$ 被放大回 $10^{-5}$，然后 $v$ 被重新抬高，如此反复。这种尖峰在精确算术中也会出现；但它是非线性的，会把 $10^{-13}$ 量级的舍入差异放大，所以两条本应重合的轨迹在尖峰阶段失去相关性。1,700 次 Adam 评估中有 678 次在达到阈值后又回升到 $10^{-6}$ 以上，128 次在最终步不再满足阈值。

因此对 Adam 两种方法，晚期检查点上的 $\Delta_B$ 混合了坐标依赖和收敛后尖峰的混沌放大。设计里已有的对照正好给出这一部分的噪声水平：特征坐标 Adam 和 $90^\circ$ 的 Adam 配对在精确算术中 $\Delta_B=0$，它们实测的 $\Delta_B$ 分布可以作为比较普通 Adam 的参照。首次达到时间 $T$ 与 1,000 步检查点不受影响。这一点也不能解读为 P4 失效：分离只发生在已经达到阈值之后的尖峰阶段。

## 10. 交接

| 对接对象 | 交给对方 | 需要对方确认 |
| --- | --- | --- |
| 第 1、2、5 部分 | 主实验对各算法的调用方式（只用公共 `RunConfig` 字段） | 算法参数名和默认值不再变化；若要改，先开 issue |
| 第 4 部分 | 上表中的整洁结果、`cluster_id`、配对表和失败日志；smoke 输出可先用于开发 | 统计单位、$\Delta_B$ 与成功表的定义 |
| 第 5 部分（整合） | 英文第 4 节与第 5.1–5.2 节 | 与总报告符号一致 |
| 全组 | §5 中两点协议细化 | 是否保留默认平分规则 |

## 11. 一分钟说法

> 我负责实验框架：把所有问题、旋转和初始点都做成可以从 seed 单独重建的配对实例，pilot、调参、评估三套 seed 互不重叠。每个算法 16 个候选学习率，GD 和 heavy ball 只在稳定区间内搜索；pilot 发现最优在边界上就扩展范围，最多两次，然后冻结。调参有 transfer 和 retune 两种，评估跑 50,000 次梯度调用，失败、截断和正常结束分开记录，结果整理成第 4 部分可以直接用的配对表。所有选择都能由另一套代码从原始记录重新算出来。
