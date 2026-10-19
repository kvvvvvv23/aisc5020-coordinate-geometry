# 第三部分：实验框架、调参与批量运行

负责人：CHEN Yuchan（@Sans）。本模块把 proposal 第 4–5 节的实验设计变成可以复现、可以中断续跑的代码：问题与配对初始点的生成、pilot、两种调参方式、批量评估和失败日志，并把整洁的结果表交给第 4 部分做统计。英文报告段落见 [part3_report.tex](part3_report.tex)。

**边界**：本模块不实现、不修改任何优化器，所有运行都通过公共的 `run(problem, initial, RunConfig)`；也不做 bootstrap、成功率区间等统计推断和结果图（第 4 部分负责）。smoke 和 pilot 的数字都不是研究结论。

## 1. 文件一览

| 文件 | 内容 |
| --- | --- |
| `src/coordinate_geometry/part3_design.py` | 谱、旋转、配对初始点、seed key、各阶段的运行单元 |
| `src/coordinate_geometry/part3_tuning.py` | 候选网格、稳定区间上限、评分、选择规则、pilot 范围调整 |
| `src/coordinate_geometry/part3_experiments.py` | pilot / tune / evaluate / report 四个阶段、并行与续跑、输出表 |
| `src/coordinate_geometry/verify_part3.py` | 从 seed 独立重建、抽样复跑、重算选择，并检查协议、完整性与执行来源 |
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

1. **边界的判定**。在默认的较小学习率平分规则下，只有边界候选得分**严格优于**相邻候选时才算边界最优；边界开始的同分平台记为 `edge_plateau`，不扩展。这是 pilot 的操作规则，不是对有限网格全局最优性的证明。如果同分确实来自 $10^{-16}$ 下限，那么没有候选能改善已到 $-16$ 的截断评分；但未截断损失、首次达到时间及 50,000 步后的表现仍可能不同。
2. **平分规则对达到时间的影响**。"取较小学习率"选择的是同分候选中的最小学习率，不保证最小化首次达到时间 $T$，也不保证同分候选具有相同的原始损失或晚期精度。代码保留此默认规则，同时提供 `tuning.tie_break: "earliest_attainment"`（平分时先比中位达到时间，再取较小学习率）。切换规则需要在新目录重新 pilot，并在报告中说明。

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

`pairs.csv` 只把数据配好，不计算区间；$\theta=0^\circ$ 的对角参考本身不再作为一对出现。`budget_summary.json` 记录各阶段梯度调用数和 5,000 / 50,000 两个时间范围的差别；新版还记录 `active_studies` 与 `report_scope`，说明当前整研究子集的范围，不把已删研究的缓存计入当前表格。新格式的 `execution_origin.json` 记录首次执行的配置、规范化 LF 源码哈希、环境和来源指纹；`manifest.json` 的执行身份来自该记录，并保留阶段耗时，不能把报告时的当前源码冒充历史执行源码。

## 8. 运行与复核

```powershell
.\scripts\part3.ps1                         # 测试 + smoke
.\scripts\part3.ps1 -Stage all -Workers 6   # 主实验：pilot → tune → evaluate → report → 独立复核
```

```text
python -m coordinate_geometry.part3_experiments --config configs/part3.json --output outputs/part3 --stage all --workers 6 --plot
python -m coordinate_geometry.verify_part3 --output outputs/part3 --sample 60
```

- 新格式首次 pilot / all 用排他创建写入 `execution_origin.json`（版本 2），后续不会覆盖；每条新 raw 记录有 `execution_source_sha256`。续跑先校验来源、Python / NumPy 版本、配置和 run 规格，再跳过已完成运行。前置 guard 拒绝时不改写 `run_config.json`、`manifest.json` 或旧验证标记；guard 通过后开始新执行阶段会移除旧 `verification.json`，要求完成后重新核验。
- `studies` 只允许删掉**整项研究**，保留下来的研究定义必须与初始配置一致。维度、条件数组、初值归一化或尺度、角度 / 旋转计划和样本数都不能在同目录修改；评估预算、检查点及其他配置也保持一致。需要改变分布或协议时，使用新目录重新 pilot。
- 旧 Release 没有 version-2 origin，不能直接作为 pilot / tune / evaluate / report 的执行或续跑目录，也不能补写一个当前来源冒充历史来源。核验历史数据时应使用 `--read-only`，以保留所有原工件（包括原 `verification.json`）；保存的旧数据继续作为 §9 的历史结果。
- 断点续跑会先移除 JSONL 末尾未写完、且没有行尾换行的片段，再追加新记录；若最后一条记录合法但没有换行，只补分隔，保留该记录。有完整行尾的坏行或文件中间的损坏记录拒绝继续，不静默跳过。配置中的 seed、维度和各类计数 / 步数要求整数；不接受布尔值或将小数截断成整数。$\kappa$ 沿用原 seed namespace 的限制：有限且为不小于 1 的整数值，`10` 和 `10.0` 都接受；这不是二次型数学本身的限制。
- report 入口会先核对当前运行计划的完整性，并按兼容的整研究子集筛选输出，防止把未完成数据或不在当前计划中的旧研究混入表格。
- 并行 worker 使用 `spawn`，并设置 BLAS 单线程。多 worker 时每次 run 的计时彼此竞争 CPU，只能作参考，不能当作成本测量。
- `verify_part3` 不使用 `part3_design` 重建实例；它抽样复跑、独立重算选择，并核对 seed 阶段隔离、配对共享 $z_0$、transfer 同学习率、失败记录，以及新增的协议、完整性和来源约束。跨平台数值复跑是否一致，应以实际验证输出为准；晚期 Adam 对舍入敏感，不能仅因保存了一个旧 `passed` 就声称当前环境逐位重现。
- 增强校验先独立核对全部计划、记录、候选、选择、CSV / 配对字段和预算，再执行数值抽样；结果分别给出 `structural_checks_passed` 和 `numerical_checks_passed`，两部分都通过才记 `passed`。默认数值容差为 `rtol=1e-8`、`atol=1e-12`，记录在验证结果中，不承诺跨平台逐位一致。正常写入模式失败时会以 `passed:false` 清掉旧通过标记；添加 `--read-only` 连 `verification.json` 都不写，且 verifier 不重写 manifest。旧 Release 可在该模式下核验，不要求其历史源码等于当前 checkout，也不为它补造新格式来源。

### 本次修复验证的范围

最终集成联检通过全部 89 项测试，其中第 3 部分为原流程 18 项、来源与恢复 guard 11 项、独立 verifier 15 项，共 44 项；覆盖 strict inputs、整研究子集、report 完整性和独立核验。各模块默认实验也通过联检。Part3 smoke 核验通过全部结构检查（168 次 pilot、600 次调参、240 次评估）并每阶段抽样复跑 60 次。这些默认 / smoke 检查不等于重新执行完整主实验。

原 Release 的只读核验通过全部结构检查（5,760 次 pilot、52,000 次调参、10,200 次评估、6,600 对），但**每阶段 150 次、共 450 次的数值复跑没有全部通过**。按 `rtol=1e-8`、`atol=1e-12`，56 次运行超出容差：pilot 7 次、tuning 11 次、evaluation 38 次；算法上是 Adam 23 次、特征坐标 Adam 33 次，其余抽样运行通过。共有 215 个字段比较失配，不能写成 215 次运行失败；其中一条 pilot 的首次达到时间也不同。Adam / eigenbasis Adam / runner / problems 四个核心文件与历史 `575ebf4` 一致。这次 450 样本核验发生在最后一次仅涉及 CSV 元数据约束的 verifier 补丁之前。

最终 checker 又以 `--sample 1 --read-only` 单独补验，通过同一历史数据的全部结构检查，并每阶段复跑 1 次、共 3 次数值样本。这个小抽样通过不能替代此前 450 次抽样中的失配，也不能把两次检查混写成一条“最终命令全部通过”的证据。两步范围、源码身份、失配汇总和受控例见[审计摘要 JSON](part3_validation_2026-10-19.json)。

受控单例中，仅将 $w_0$ 第一分量改变 1 ULP（扰动范数 $2.78\times10^{-17}$），终点相对损失就从 $9.46\times10^{-5}$ 变为 $1.08\times10^{-7}$，并改变最终是否保持阈值；该受控例的首次达到时间保持为 122。这支持数值敏感性，但不能证明历史差异究竟由哪一次舍入或平台差异引起。原数据可以作为已核验结构的历史运行，不能称为在当前环境下数值完全复现。

## 9. 原 Release 主实验运行（2026-10-18/19，commit `575ebf4`）

以下数字来自同学的 [part3-v1 Release](https://github.com/akaComic/aisc5020-coordinate-geometry/releases/tag/part3-v1)，是在原版本 `575ebf4` 上跑完的一版完整数据，供第 4 部分开发和组内检查。它不是审查修复后重新运行的完整主实验。如果修改协议（例如 §5 的平分规则、谱、维度或初值分布），需要在新目录重新 pilot 和运行。

原 manifest 的环境为 Linux、Python 3.13.16、NumPy 2.5.4，2 个 worker，BLAS 线程环境变量为 1。

| 阶段 | 运行数 | 梯度调用 | 2 个 worker 的墙钟时间 |
| --- | --- | --- | --- |
| pilot | 5,760 | 2.85 × 10⁷ | 3.4 分钟 |
| 调参：transfer / retune | 52,000 | 0.78 × 10⁸ / 1.76 × 10⁸ | 57 分钟 |
| 评估：transfer / retune / theory | 10,200 | 2.13 × 10⁸ / 2.13 × 10⁸ / 0.85 × 10⁸ | 约 1 小时 51 分钟（中途按设计续跑一次） |

- 500 个调参格子全部选出学习率，没有 tuning failure；47 个最优在网格上端，全部是 GD / heavy ball 落在稳定上限处（`stability_capped`），没有范围限制；40 个 heavy ball 格子（$\kappa=1,10$）是 `edge_plateau`。
- 1,261 条非 `finite_cap` 记录全部来自次要方法 `diagonal_gd` 在 pilot 和调参中的大学习率候选（`explosive`），这些候选按规则记最差分。10,200 次评估运行全部是 `finite_cap`，所有配对双方都达到过 $10^{-6}$。
- 原 Release 的 `verification.json` 保存了每阶段抽 150 次 run 的 `passed` 结果；原报告说明 500 个选择独立重算一致、6,600 对配对检查通过。这是原版本的历史复核记录；新修复版本的验证范围另行说明。
- 本次只读审查另核对了 Release ZIP 与 GitHub asset 的 SHA256、历史快照的 19 个源码哈希，并独立检查全部 6,600 对的初始 seed / 元数据配对及检查点 $\Delta_B$，最大重算差为 $1.78\times10^{-15}$。这是工件完整性检查；另行完成的 450 次数值复跑及其失配见 §8，均不是重新执行完整主实验。

### 给第 4、5 部分：Adam 在收敛后的尖峰与配对分离

`pairs.csv` 新增两列：`correspondence_expected`（精确算术下这一对应当逐步重合：GD、heavy ball、特征坐标 Adam 在同一设置下总是如此；Adam 与对角预条件只在带符号置换时如此）和 `separation_step`（两条记录轨迹的 $\log_{10} r$ 第一次相差超过 $10^{-6}$ 的记录步）。

先限定为 `correspondence_expected=True` 的子集，即数学对应性需要的算法设置一致。原 Release 中共有 4,460 对；其保存的轨迹在双方首次达到阈值前，记录点上都没有分离。**这个子集内**的 GD、heavy ball 和对角预条件始终不分离（$|\Delta_B|\le3\times10^{-10}$）。该结论不能扩大到全部 retune 配对：它们可能选到不同学习率，产生正常的轨迹差异。对应性子集中，1,118 个 Adam 或特征坐标 Adam 配对后来分离，中位记录分离步为 17,000；晚期 $\Delta_B$ 可能很大。

1,700 次普通 Adam 评估中，有 678 次在首次达到阈值后，**于保存的记录点上**被观察到重新超过 $10^{-6}$；128 次最终不满足阈值。记录间隔为 1,000 步，678 不是对所有中间步回升事件的完整计数。原 Release 保存的对应性配对中，双方首次达到时间 $T$ 全部相同，1,000 步的绝对配对 log gap 不超过 $3\times10^{-10}$；这是同一套历史数据内的观察，不保证跨环境复跑的首次达到时间相同，见 §8 的实际失配。

**解释尖峰时须保留一阶动量和 epsilon。** 取一维二次型的曲率 $\lambda>0$，只在晚期忽略 bias correction，并暂时冻结分母 $c=\sqrt{\bar v}+\epsilon>0$，令 $\gamma=\eta/c$。这个局部线性近似为

```math
m_{t+1}=\beta_1m_t+(1-\beta_1)\lambda x_t,\qquad
x_{t+1}=x_t-\gamma m_{t+1}.
```

消去一阶矩后：

```math
x_{t+1}=[1+\beta_1-\gamma(1-\beta_1)\lambda]x_t-\beta_1x_{t-1},\qquad
0<\gamma\lambda<\frac{2(1+\beta_1)}{1-\beta_1}.
```

当 $0\leq\beta_1<1$ 时，右侧是该**冻结分母模型**的严格 Schur 稳定条件。$\beta_1=0.9$ 时上界是 38，而不是 GD 的 2。例如 $\gamma\lambda=3$ 时根为 $0.8\pm0.509902\,\mathrm i$，模为 $\sqrt{0.9}<1$，仍稳定；38 处有根 $-1$，不属于严格稳定。实际 Adam 分母为 $\sqrt{\hat v_t}+\epsilon$，有效尺度上限是 $\eta/\epsilon$。

当梯度平方对二阶矩的补充很小时，二阶矩衰减可能提高有效尺度，为后续回升提供一种解释；但真实 Adam 的分母在变化，38 不能作为完整自适应系统的全局稳定阈值。一次 double 复跑不能证明同样尖峰在精确算术中必然出现，也不能证明数学意义的混沌。现有轨迹支持的是晚期损失回升和数值敏感性观察。

因此，晚期 $\Delta_B$ 需要同时考虑坐标、调参和数值敏感性。算法设置匹配的特征坐标 Adam 与带符号置换对照可用于诊断后者；不同方法的敏感性分布不一定相同，不能直接把它当成可扣除的统一噪声。观察到的浮点分离不反驳 P4 的精确算术共同基恒等式。

## 10. 交接

| 对接对象 | 交给对方 | 需要对方确认 |
| --- | --- | --- |
| 第 1、2、5 部分 | 主实验对各算法的调用方式（只用公共 `RunConfig` 字段） | 算法参数名和默认值不再变化；若要改，先开 issue |
| 第 4 部分 | 上表中的整洁结果、`cluster_id`、配对表和失败日志；smoke 输出可先用于开发 | 统计单位、$\Delta_B$ 与成功表的定义 |
| 第 5 部分（整合） | 英文第 4 节与第 5.1–5.2 节 | 与总报告符号一致 |
| 全组 | §5 中两点协议细化 | 是否保留默认平分规则 |

## 11. 一分钟说法

> 我负责实验框架：把所有问题、旋转和初始点都做成可以从 seed 单独重建的配对实例，pilot、调参、评估三套 seed 互不重叠。每个算法 16 个候选学习率，GD 和 heavy ball 只在稳定区间内搜索；pilot 发现最优在边界上就扩展范围，最多两次，然后冻结。调参有 transfer 和 retune 两种，评估跑 50,000 次梯度调用，失败、截断和正常结束分开记录，结果整理成第 4 部分可以直接用的配对表。所有选择都能由另一套代码从原始记录重新算出来。
