# Same Spectrum, Different Optimization

仓库：https://github.com/kvvvvvv23/aisc5020-coordinate-geometry

完整第 5 部分见 [中文交接指南](docs/part5_guide.md) 和 [英文 LaTeX 报告源文件](docs/part5_report.tex)。[Module 5 Release](https://github.com/kvvvvvv23/aisc5020-coordinate-geometry/releases/tag/module5-v1) 提供报告 PDF、原始验证结果和完整交付 ZIP。协作及恢复操作见 [GitHub 使用说明](docs/upload_to_github.md)。

AISC5020 小组项目：在固定 Hessian 特征值的条件下，研究坐标旋转如何影响 GD、heavy-ball momentum、Adam 和预条件方法。

这个仓库已接入第 1、2、3、5 部分：统一的问题与运行接口、GD / heavy-ball 理论验证、Adam 坐标依赖诊断、调参与批量实验框架，以及特征坐标 Adam 独立验证。第 3 部分已发布按现有协议运行的一版完整实验数据；全组最终结论仍需组内确认协议，并由第 4 部分对接统计分析。模块 smoke 和结构验证不能替代主实验。

## 分工

| 部分 | 负责人 | 主要交付 |
| --- | --- | --- |
| 1 | CHENG Zixuan | GD / heavy-ball 推导、实现与理论基线验证 |
| 2 | 坤 | Adam 实现、旋转反例与坐标依赖诊断 |
| 3 | @Sans | 问题生成、配对初始化、调参与批量运行 |
| 4 | HUANG Ze | 对角预条件、有效条件数与统计分析 |
| 5 | HOU Kaiwen | 特征坐标 Adam、复现与报告整合 |

每位负责人同时提交本部分的推导、代码、验证结果和英文报告段落。编号表示模块，不表示工作量或先后顺序。详细依赖与进度见 [协作计划](docs/team_plan.md)。

## 先跑起来

在仓库根目录执行。以下是 PowerShell 示例，先安装 Python 3.10+ 和 Git。基础运行仅依赖 NumPy；依赖会随项目一同安装。

```powershell
.\scripts\setup.ps1
.\scripts\check.ps1
```

`setup.ps1` 在 `work/.venv` 建立环境并安装项目；`check.ps1` 运行数学验证和 smoke，并把缓存与临时文件保存在项目 `work/` 内。若需要指定 Python，可以使用 `-PythonExecutable` 参数。

手动运行时，使用环境中的 Python：

```powershell
.\work\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\work\.venv\Scripts\python.exe -m coordinate_geometry.smoke --config configs/smoke.json --output outputs/smoke
```

macOS / Linux 可以运行 `python -m venv work/.venv`，再执行 `work/.venv/bin/python -m pip install -e .`；上方命令中的 Python 路径替换为 `work/.venv/bin/python`。已激活项目环境时，也可以直接使用 `python`。

本地输出保存在 `outputs/`。执行成功说明基础流程能运行；不能据此声称某个方法更好或主实验已经完成。提交实验分析时，应附运行命令、配置、seed、代码版本和输出说明。

已接入的模块可以一次检查：

```powershell
.\scripts\verify_all.ps1 -NoPlots
```

该命令运行全部测试、公共 smoke、第 1、2、5 部分的默认实验，以及第 3 部分的小预算端到端实验和完整性核验。它不启动第 3 部分耗时较长的主实验。需要同时生成图表时，先安装 `.\work\.venv\Scripts\python.exe -m pip install -e '.[plots]'`，然后去掉 `-NoPlots`。GitHub Actions 在 Python 3.10 和 3.12 上执行相同的数值检查。报告的 `.tex` 源文件保留在 `docs/`；编译 PDF 和中间文件放在 `outputs/` 或 `work/`。

安装提供 `pdflatex` 的 MiKTeX 或 TeX Live 后，可运行 `.\scripts\build_reports.ps1`，双次编译第 1、2、3、5 部分报告到 `outputs/reports/`；脚本会检查未解析引用及内容溢出。

## 第 2 部分：Adam 坐标依赖诊断

第 2 部分使用固定的 canonical GD 梯度历史，将它及其旋转版本分别送入从零开始的
Adam moment accumulators。这样得到的 replay defect 隔离了逐坐标归一化本身，不会混入
Adam 轨迹已经分叉后的梯度差异。诊断还记录普通 Adam 在匹配初始点下的逐步位置对应误差。

运行数值诊断：

```powershell
.\scripts\adam_diagnostics.ps1
```

需要生成 `defect_versus_angle.png` 时，先安装绘图依赖，再运行：

```powershell
.\work\.venv\Scripts\python.exe -m pip install -e '.[plots]'
.\scripts\adam_diagnostics.ps1 -Plot
```

配置保存在 `configs/adam_diagnostics.json`，结果写入 `outputs/adam_diagnostics/`，包括每个
角度和步数的 numerator、denominator、relative defect、普通 Adam 位置误差、运行配置和
汇总信息。这是机制诊断，不是收敛界、调参结果或优化器优劣排名。

## 统一接口

问题使用严格凸二次型：

$$
f(w)=\tfrac12 w^\top H w,\qquad H=QDQ^\top,\qquad D\succ0,\quad Q^\top Q=I.
$$

`QuadraticProblem` 保存特征值向量 `eigenvalues`、旋转矩阵 `rotation` 和 Hessian `hessian`，提供 `.loss(w)`、`.gradient(w)` 和 `.dimension`。各模块共用 `run(problem, initial, RunConfig(...))`，返回 `RunResult`。

| 约定 | 内容 |
| --- | --- |
| 配置 | `algorithm`、`learning_rate`、`max_steps` 及算法专属参数 |
| 轨迹 | `step`、`loss`、`relative_loss`、`gradient_norm_before_step` |
| 结果 | `status`、`first_attainment`、`threshold_retained`、`gradient_calls` 和元数据 |
| 步数 | `step=0` 是更新前的初始点 |
| 配对 | 同一实例中的不同方法共用问题和初始点；旋转前后使用对应的初始点 |

新增算法时保留这些记录，不要为自己的一组实验另起一套不兼容的格式。更改公共接口前先开 issue，给依赖模块留出同步机会。具体类和字段以 `src/coordinate_geometry/` 中的代码为准。

算法标识为 `gd`、`heavy_ball`、`adam`、`eigenbasis_adam`、`diagonal_gd`。最小调用示例：

```python
import numpy as np
from coordinate_geometry import QuadraticProblem, RunConfig, run

problem = QuadraticProblem(
    eigenvalues=np.array([1.0, 10.0]), rotation=np.eye(2)
)
result = run(
    problem,
    initial=np.array([1.0, 1.0]),
    config=RunConfig(algorithm="gd", learning_rate=0.1, max_steps=100),
)
print(result.status, result.first_attainment)
```

运行器完成固定预算后，`finite_cap` 表示预算内没有触发数值失败或爆炸阈值；是否到过目标阈值看 `first_attainment`。当前 `threshold_retained` 表示完整预算结束时仍位于阈值内，不代表从首次达到后一直没有回升。不要仅凭 `status` 判断成功或失败。轨迹中的梯度范数对应这一步更新前的梯度，初始记录为 `None`；损失对应更新后的点。

## 协作方式

`main` 保留能通过测试的版本。第一次参与请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)：**公开仓库链接可以直接查看和 fork；点击链接不会自动获得原仓库写入权限**。没有写入权限的成员使用 fork → 分支 → Pull Request，同样可以协作。

建议每个人先提交一个可运行的小例子，确认接口后再扩展实验。公共代码、配置和说明走 PR；大体积结果、课程原始材料和包含学号的文件不提交到公开仓库。

## 本周检查点

- 2026-10-18（周日）前：各模块提交可运行的小例子、推导进展、当前阻塞和下一步。
- 2026-10-19 至 10-25：争取连通核心流程、完成 pilot 并开展主实验。这个时间安排是组内目标，需要根据 pilot 成本调整。

## 结果使用原则

- 参数调节、主实验和演示运行分别记录，不能把调参表现当作独立验证结果。
- 未达到阈值、数值失败和正常结束分别记录；比较时报告失败与截断情况。
- 预条件与特征坐标方法说明额外信息和计算成本，不只比较更新步数。
- 课程报告中的贡献说明以实际提交和完成的工作为准。

## 第 1 部分：GD / heavy-ball 理论基线

完整推导、二维手算、实测结果与模块交接见 [第一部分中文指南](docs/part1_guide.md)，英文报告见 [LaTeX 源文件](docs/part1_report.tex)。使用已激活的项目环境，在仓库根目录执行：

```bash
python -m unittest discover -s tests -v
python -m coordinate_geometry.part1_experiments --config configs/part1.json --output outputs/part1
python -m coordinate_geometry.verify_part1 --output outputs/part1
```

实验覆盖 7 组理论参数 / 稳定边界 / 发散对照和 4 个旋转角度，保存 56 条完整轨迹、参数、状态、阈值记录、源码哈希与图表。独立复核程序使用特征方向的二阶位置递推重建结果。只需数值输出时，在实验命令末尾加 `--no-plots`。输出写入 `outputs/part1/`，再次运行会更新同目录结果。

这些结果用于验证旋转对应性、GD 收敛界和 heavy-ball 稳定性，不是全组主实验。heavy-ball 的谱半径是渐近状态因子，重根暂态可以使损失先增大；不能将其直接当作每步目标函数收缩界。

## 第 2 部分：Adam 坐标依赖诊断

实现与读代码说明见 [中文指南](docs/part2_guide.md) 和 [梯度回放说明](docs/adam_replay_comparison.md)；公式、正 epsilon 反例、符号排列对应性和英文方法段落见 [LaTeX 报告](docs/part2_report.tex)。

```text
python -m coordinate_geometry.adam_diagnostics --config configs/adam_diagnostics.json --output outputs/adam_diagnostics
```

默认在 7 个旋转角度各运行 50 步，保存 350 行诊断。梯度回放使用同一段 canonical GD 梯度历史，隔离 Adam 逐坐标归一化的影响；另行记录普通 Adam 实际轨迹的对应误差。两者含义不同，均不能直接解释为收敛界或优化器排名。`max_steps`、`seed`、`checkpoints` 必须使用整数，不能用小数或字符串代替。需要图表时添加 `--plot`。

## 第 5 部分：特征坐标 Adam

本模块的学习与交接说明在 [part5_guide.md](docs/part5_guide.md)，英文理论、证明与解释在 [part5_report.tex](docs/part5_report.tex)。实现支持生成基 `oracle`、仅从 Hessian 恢复的 `eigh` 基，以及验证后的 `supplied` 基；固定基在优化开始前准备，可显式复用。

单独运行本模块测试：

```powershell
.\work\.venv\Scripts\python.exe -m unittest discover -s tests -p test_eigenbasis.py -v
```

需要生成本模块验证数据和图表时，先安装绘图依赖，再运行：

```powershell
.\work\.venv\Scripts\python.exe -m pip install -e '.[plots]'
.\scripts\part5.ps1
```

`part5.ps1` 先运行测试，再读取 `configs/part5.json` 生成验证数据与图表，最后由另一套代码重建 seed 对应的问题和初始点，独立重算保存的 Adam 轨迹、损失、汇总、基残差和计时中位数。只需要数值输出时，可使用 `.\scripts\part5.ps1 -NoPlots`，不要求绘图依赖。完整实验可能覆盖同目录旧输出，需保留旧版本时先另存。

跨平台或手动运行：

```text
python -m coordinate_geometry.part5_experiments --config configs/part5.json --output outputs/part5
python -m coordinate_geometry.verify_part5 --output outputs/part5
```

结果写入 `outputs/part5/`，包括原始轨迹、配置、对应性 CSV、特征基诊断、计时、汇总、图表和独立复核记录。本模块验证的是共同特征基中的轨迹对应性及其边界，报告基恢复残差、轨迹误差和额外成本；它不是全组调参后的优劣结论。特别是重特征值允许任意子空间换基；极小特征值间隔也会影响浮点恢复，因此不能仅凭“特征值数值上互不相同”或残差小，就宣称恢复轨迹一定一致。

## 第 3 部分：实验框架、调参与批量运行

完整说明见 [第三部分中文指南](docs/part3_guide.md)，英文报告段落（第 4 节与第 5 节调参部分）见 [LaTeX 源文件](docs/part3_report.tex)。本模块只通过统一的 `run(problem, initial, RunConfig)` 调用各模块的优化器，不改动其他模块的算法实现。

先用几秒钟的 smoke 检查整条流程和输出格式（不是 pilot，也不是调参结果）：

```powershell
.\scripts\part3.ps1                      # 测试 + smoke，输出到 outputs/part3_smoke
```

主实验按阶段运行；仅在代码来源和协议一致时才复用已保存的运行：

```powershell
.\scripts\part3.ps1 -Stage pilot         # pilot：确定并冻结各算法的 16 个候选学习率
.\scripts\part3.ps1 -Stage tune          # transfer 与 retune 两种调参
.\scripts\part3.ps1 -Stage evaluate      # 50,000 次梯度调用的评估运行与理论参考
.\scripts\part3.ps1 -Stage report        # 整理表格并独立复核
```

跨平台或手动运行：

```text
python -m coordinate_geometry.part3_experiments --config configs/part3.json --output outputs/part3 --stage all --workers 4 --plot
python -m coordinate_geometry.verify_part3 --output outputs/part3
```

结果写入 `outputs/part3/`：`raw/` 保存每次运行的完整记录（JSONL），`tables/` 提供给第 4 部分的整洁表格（`evaluation_runs.csv`、`pairs.csv`、`evaluation_traces.csv`、`failures.jsonl`、`selected_rates.csv`、`tuning_scores.csv`、`status_summary.csv`），另有 `execution_origin.json`、`frozen_protocol.json`、`budget_summary.json` 与 `manifest.json`。pilot、调参与评估使用互不重叠的 seed 流。来源指纹、研究定义或预算不兼容时会拒绝续跑及重新生成报告，需使用新目录；失败的校验不会覆盖旧来源。旧版无来源文件的实验可独立核验，但不会被自动补记成新实现的结果。

本次修复及验证的实际范围见 [审计记录](docs/part3_validation_2026-10-19.json)。当前默认联检通过；原 Release 数据的完整性与表格核验通过，但在当前 Windows 环境复跑的 450 个历史样本中，56 个 Adam / 特征坐标 Adam 运行超出数值容差，不能标记为完整的跨平台数值复现通过。原数据及来源保持不变，具体限制见第三部分指南与报告。
