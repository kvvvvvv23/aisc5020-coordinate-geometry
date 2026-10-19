# Part 4 最终交接说明

负责人：HUANG Ze。本交付是模块 4，不是替全组完成或提交最终总报告。

## 贡献来源与团队整合版本

原始模块来自 [acbldppx/aisc5020-part4-huangze](https://github.com/acbldppx/aisc5020-part4-huangze)，快照提交为 `1d9b405e0626e2c9b24a2fcd165025f2e68b97df`，保留 HUANG Ze 署名。团队导入的是经过隐私清理的文件快照，并在其上修复审查发现的问题；没有合并上游 Git 历史。

团队版本来自 [kvvvvvv23/aisc5020-coordinate-geometry](https://github.com/kvvvvvv23/aisc5020-coordinate-geometry) 的 `main`；[整合版 Release](https://github.com/kvvvvvv23/aisc5020-coordinate-geometry/releases/tag/part4-integrated-v1) 的 `part4-integrated-v1` 标签固定评审后的整合源码。本次清理针对团队导入副本和新生成的发布产物；贡献者原仓库、原 Release 和历史附件未由本次整合修改。

## 交付范围

1. P5 推导、二维有效谱/条件数、稳定区间、eta=1 的 minimax 步长。
2. 沿用公共 preconditioning.py / runner.py，不修改其他成员的算法定义；新增独立 P5 实验和测试。
3. 对历史正式输出的 10,200 个评估 run、6,600 对进行只读核对。
4. 首次达到和最终保留两种四格成功表、独立 run 的分母统计、失败计数、固定预算 log-gap、配对效应、达到时间比。
5. 二维联合 initial-ID / 高维 rotation-cluster bootstrap，2,000 次，固定 seed，描述性点态 95% 区间。
6. 条件数/收敛图、二维 gap maps、角度效应图、高维区间图、成功表/保留图及失败图。
7. 英文模块报告 PDF/LaTeX、中文说明、可运行源码、测试、原始结果与来源哈希。

## 不可省略的限制

- 正式数据是 commit `575ebf4` 的历史结果；贡献者分析时采用的团队源码基线是 `bb365b9`，模块来源快照是上述 `1d9b405`。这些身份与后续团队整合提交分别记录。统计是对保存结果的分析，不是把当前源码当成历史执行源码。
- 贡献者保存的 verifier 审计中，全部结构核验通过；每阶段 10 个、共 30 个数值复跑中，2 个 eigenbasis-Adam run 的 9 个字段超出容差，因此 `numerical_checks_passed=false`。这次原始审计没有覆盖历史 `verification.json` 的旧 passed 标记。
- 团队前次审计另有 450 个样本中 56 个 run 失配的记录，和贡献者这次 30 个样本分开说明；不能把任何一份旧 passed 当成当前长轨迹全部复现的证据。
- 所有运行都曾达到 1e-6，不代表最后成功。Adam 的 transfer 保留 777/850，retune 795/850；总计 1572/1700。
- late-stage Adam/eigenbasis Adam 不一致不能直接推翻精确算术理论；retune 还可能改变两边学习率。
- 未声称完成独立、受控的 setup-inclusive 成本基准。零 setup_seconds 不能当作无预条件成本。该计时是 Proposal 次要扩展，报告中明确限制。
- bootstrap 只有每格 10 个独立方向/旋转，区间不是同时置信带，也没有包括跨平台不确定性；全成功的退化区间不是成功概率为 1 的证明。

## 复现（无需重新跑主实验）

在团队仓库根目录，安装项目依赖及 matplotlib：

```bash
python -m pip install -e '.[plots]'
python scripts/reproduce_part4.py
```

重建 PDF 需要 TeX Live/MiKTeX/MacTeX 的 pdflatex：

```bash
python scripts/reproduce_part4.py --report
```

先从团队整合版发布包取得历史数据，保留其目录结构和来源记录；默认输入目录为 `outputs/part3/`。没有这些输入，单独克隆源码不能重做正式统计。已有整合版 PDF 可直接阅读，没有 LaTeX 也能重做所有 CSV 和图。
复现脚本会重建 Part 4 输出，但不会改写 outputs/part3/ 历史数据。
`requirements-part4.txt` 保留原作者统计环境：Python 3.12.14、NumPy 2.3.5、matplotlib 3.10.8，不能把它当作团队本轮已使用的环境。团队整合验证使用 Python 3.12.14、NumPy 2.5.4；实际命令、依赖和结果以该次验证日志及 JSON 为准。历史 Part 3 执行环境另由其原 manifest 记录。
整合验证应运行仓库全套测试及 Part 4 复现，并保存实际日志。测试通过不替代主实验的数值复现。

## 给整合同学的交接顺序

1. 使用团队评审后的整合提交，按 `part4-integrated-v1` 标签固定版本。`bb365b9` 只标识原贡献的分析基线，不是要求回退到的当前版本。
2. 模块已按文件快照导入团队仓库，无需再次复制旧 `part4_merge/`，也不要引入上游仓库的完整历史或覆盖公共算法。
3. 大型历史 `outputs/part3/` 放入经过检查的新发布包，源码仓库保留模块源码、测试、文档和必要的小型审计记录。发布 PDF 必须从团队清理后的 TeX 重新编译，不能沿用未清理的上游附件。
4. 执行 Part 4 测试及复现命令，对照 summary CSV 和报告数字。
5. 在总报告加入 P5、统计方法、成功/保留区分、效应与 bootstrap，选择少量图，并保留数值复现限制。
6. 组员阅读并确认公式、配对结构和关键表；程序测试和自动化审查不能写成另一位同学已经独立复现。
7. 报告/代码使用了 AI 辅助整理与实现，请按课程 AI 使用规范检查、理解并披露，不要未经阅读直接以个人独立完成名义提交。

## 重要路径

- outputs/part4/pdf/part4_report.pdf：英文模块报告。
- outputs/part4/stats/：逐对数据、汇总、成功表、失败表、审计。
- outputs/part4/figures/：正式历史统计图；outputs/part4/*.png 是独立 P5 图。
- docs/part4_validation_2026-10-19.json：本轮整合检查、统计对照及输入来源核对记录。
- docs/part4_report.tex：报告正文；part4_generated_tables.tex 为真实 CSV 生成的表。
- docs/part4_release.md：团队整合版的来源与发布说明。
