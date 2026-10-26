# 我的复现仓库：https://github.com/LINXiaoyuan072/TFGFormer_reproduction.git

## TFGFormer Hardanger Bridge 小组分享版复现总结

本文用于向小组成员完整说明 `Time-frequency gated transformer for structural dynamic response reconstruction` 的 Hardanger Bridge 复现工作。内容包括研究目标、资料来源、数据预处理、训练环境、论文与发布代码差异、正式训练参数、诊断调参、评价指标、当前结果、论文对比口径、定性分析、重要文件和下一阶段计划。

阅读时请区分两类结果：`released` 表示尽量遵循作者发布网络和训练循环的复现；`channel`、梯度裁剪和固定等权集成属于为分析数值发散和周期伪峰而增加的诊断实验。诊断结果不能替代论文原配置结果。

本文档总结 `Time-frequency gated transformer for structural dynamic response reconstruction`
在 Hardanger Bridge 数据上的当前复现过程、关键脚本、可调参数、已有结果以及与论文的比较方法。

## 1. 当前复现结论

本工程已经完成以下工作：

1. 获取公开论文、补充材料、官方 Source Data 和作者 v1.0.0 代码。
2. 获取 Hardanger Bridge 官方 `MidFreq_2014_10.zip`，完成校验、解析、通道排序、切窗、随机划分和三种固定缺失场景构造。
3. 建立论文版本环境：Python 3.10.22、PyTorch 2.3.0+cu121、cuDNN 8.9.2。
4. 在学习率 0.01、batch size 16、200 epoch 下运行作者发布网络，并恢复发布脚本每轮重置 seed 42 的行为。
5. 完成 S25 和 S75 发布布局训练；发布布局的 S50 在两个种子下早期数值发散。
6. 完成 channel-layout 门控和梯度裁剪诊断，得到稳定的 S50、S75 结果。
7. 输出全局指标、仅缺失位置指标、零填充基线、波形、频谱、残差、误差热图和三场景汇总图。

当前属于**单月子集上的方法和流程复现**，还不是论文表格的完整复现。最主要的原因是论文没有公开训练所用月份和精确窗口清单，而本工程目前只使用 2014 年 10 月的 7 个记录、119 个窗口。官方共有 58 个 MidFreq 压缩包，总计约 75.8 GiB。

## 2. 方法与数据接口

TFGFormer 输入不完整的多传感器振动响应，结合以下模块重建完整响应：

- MSDF：多尺度扩张 Transformer，提取长距离、全局时间依赖。
- Dense block：融合局部多层特征，保留瞬态和局部波形。
- Frequency Module：通过 FFT 复数变换显式建模频域特征。
- Gate Module：自适应调制、融合时域和频域信息。
- JointTimeFreqLoss：同时约束时域均方误差与频域复数谱误差。

Hardanger 单个标签文件形状为 `(16, 1024)`，训练批次形状为
`(B, 1, 16, 1024)`。定义：

- 完整真实响应：`Y`
- 观测掩码：`M`，1 为已知，0 为缺失
- 模型输入：`X = M * Y`
- 网络原始输出：`P`
- 最终重建：`M * X + (1-M) * P`

适配器使用显式掩码，所以真实测量值恰好为 0 时不会被误认为缺失；所有已知位置在最终预测中逐点保持不变。

## 3. 数据处理过程

当前原始数据：`data/raw/MidFreq_2014_10.zip`。

处理设置：

| 项目 | 当前设置 | 与论文关系 |
|---|---:|---|
| 采样率 | 10 Hz | 补充材料设置 |
| 分量 | 竖向 Z 加速度 | 补充材料设置 |
| 单位 | g | 官方数据单位 |
| 传感器 | 桥面 16 路 | 论文 Hardanger 输入 |
| 初始裁剪 | 每个独立记录去掉前 867 点 | 数值来自补充材料；作用域未明确 |
| 窗长 | 1024 点，即 102.4 秒 | 补充材料设置 |
| 步长 | 1024 点，无重叠 | 补充材料设置 |
| 额外滤波 | 无 | 补充材料设置 |
| 额外归一化 | 无 | 补充材料设置 |
| 划分 | 窗口随机 7:2:1 | 补充材料设置 |
| 划分 seed | 42 | 本工程的可复现选择 |
| 数量 | train 83 / val 23 / test 13 | 当前单月子集 |

固定缺失通道编号从 1 开始：

- S25：`2, 8, 11, 13`
- S50：`2, 5, 7, 8, 9, 11, 13, 14`
- S75：`1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 16`

传感器到模型通道的顺序根据论文图 4b、官方传感器名称和坐标推断，尚未得到作者确认。完整映射和每个窗口的原始记录、起止下标都保存在 `manifest.json` 中。

重新预处理时使用新输出目录，脚本会拒绝覆盖已有划分：

```bash
cd /zeron-vepfs/tjqc/xiaoyuan.lin/TFGFormer_reproduction
.venv/bin/python scripts/prepare_hardanger.py \
  --archive data/raw/MidFreq_2014_10.zip \
  --output data/processed/hardanger_2014_10_fig4_rebuild \
  --seed 42

.venv/bin/python scripts/check_dataset.py \
  --data data/processed/hardanger_2014_10_fig4_rebuild
```

数据检查会验证同名 feature/label/mask 配对、形状、dtype、有限值、掩码、已知值、划分互斥，以及原始区间不重叠。它不能消除同一记录中相邻窗口之间的统计相关性。

## 4. 环境

当前已验证环境：

```text
Python 3.10.22
torch 2.3.0+cu121
torchvision 0.18.0+cu121
cuDNN 8.9.2
numpy 1.26.3
scipy 1.15.3
matplotlib 3.9.0
timm 1.0.3
einops 0.8.1
GPU: NVIDIA L20
```

现有环境可直接调用：

```bash
cd /zeron-vepfs/tjqc/xiaoyuan.lin/TFGFormer_reproduction
.venv/bin/python -m pip check
.venv/bin/python scripts/smoke_model.py \
  --output results/model_smoke_new.json
```

依赖版本位于 `requirements.txt`。CUDA/cuDNN 的本地复用记录位于
`results/cuda_reuse_manifest.json`。

## 5. 严格复现训练命令

本工程使用正文学习率 0.01。作者发布 `train.py` 的默认学习率是 0.001，两者不一致。
为遵循本次任务要求，所有正式复跑都使用 0.01。

S25：

```bash
.venv/bin/python scripts/train_hardanger.py \
  --data data/processed/hardanger_2014_10_fig4/S25 \
  --epochs 200 --batch-size 16 --lr 0.01 --seed 42 \
  --gate-layout released --released-epoch-reseed \
  --output results/S25_released_new
```

S50 和 S75 只需替换 `--data` 与 `--output`：

```bash
.venv/bin/python scripts/train_hardanger.py \
  --data data/processed/hardanger_2014_10_fig4/S50 \
  --epochs 200 --batch-size 16 --lr 0.01 --seed 42 \
  --gate-layout released --released-epoch-reseed \
  --output results/S50_released_new

.venv/bin/python scripts/train_hardanger.py \
  --data data/processed/hardanger_2014_10_fig4/S75 \
  --epochs 200 --batch-size 16 --lr 0.01 --seed 42 \
  --gate-layout released --released-epoch-reseed \
  --output results/S75_released_new
```

`--released-epoch-reseed` 很关键。作者发布脚本在每轮开头调用 `seed_all(42)`，并开启 cuDNN benchmark。本适配器同时重置 DataLoader generator，使每轮训练顺序重复。没有该选项的 PyTorch 2.3 S25 实验在第 5 轮验证时出现非有限损失。

训练输出目录包含：

- `run.json`：参数、环境、代码和数据清单哈希、状态。
- `history.json`：每轮 train/val loss。
- `best.pt`：验证损失最低的权重。
- `metrics.json`：全局、缺失位置及零填充指标。
- `test_predictions.npz`：input、target、prediction、observed。
- `loss.png`：训练与验证损失曲线。
- `reconstruction.png`：固定测试窗口的时域和频域图。

若训练后期发散但 `best.pt` 已保存，可以重新评估发散前最佳权重：

```bash
.venv/bin/python scripts/evaluate_saved_best.py \
  --run results/FAILED_RUN \
  --output results/FAILED_RUN_best_eval
```

## 6. 可以调整的参数及调参方法

### 6.1 学习率 `--lr`

作用：控制 Adam 每次参数更新的步长，是当前数值稳定性最敏感的参数。

- 论文正文：0.01。
- 发布代码默认：0.001。
- 当前严格复现：0.01。
- 诊断搜索建议：`0.001, 0.003, 0.01`。

调法：固定数据划分、seed、batch size、模型和损失，只改变学习率；按最低验证损失、是否出现非有限值、最终缺失位置验证误差选择。严格论文结果必须保留 0.01，其他学习率单独标成诊断实验。

### 6.2 训练轮数 `--epochs`

作用：决定优化次数。脚本始终用验证损失最低的 `best.pt` 测试，因此增加轮数不会强制使用最后一轮。

- 论文和当前复现：200。
- 快速检查：5–20。
- 完整诊断：100、200、300。

调法：先观察 `loss.png`。如果验证损失在 200 轮前已经长期上升，应该使用最佳权重或实现早停；如果到第 200 轮仍持续下降，可诊断性地增加到 300。论文对照仍报告 200 轮设置。

### 6.3 批大小 `--batch-size`

作用：影响梯度方差、显存占用和每轮更新次数。

- 论文和当前复现：16。
- 诊断建议：8、16、32。

当前训练集有 83 个窗口，`drop_last=True`。batch 16 时每轮实际使用 80 个窗口；batch 32 时只使用 64 个，数据浪费较多；batch 8 时使用 80 个但每轮更新次数翻倍。因此小子集优先保持 16，扩充数据后再比较批大小。

### 6.4 随机种子 `--seed`

作用：控制模型初始化和初始随机状态。

- 当前主要结果：42。
- 论文报告：五次独立运行的均值和标准差。
- 建议五次种子：预先固定，例如 `0, 1, 2, 3, 4`，不可根据测试表现挑种子。

使用 `--released-epoch-reseed` 时，每轮内部顺序仍按作者行为重置为 42，但不同 `--seed` 仍能改变重置发生前的模型初始化。每个种子必须写入独立输出目录。

### 6.5 梯度裁剪 `--clip-grad`

作用：把全模型梯度范数限制在阈值内，抑制梯度爆炸。

- 发布配置：不裁剪，即不传该参数。
- 当前诊断：`--clip-grad 1.0`。
- 建议诊断范围：0.5、1.0、5.0。

当前观察：裁剪 1.0 使 S50 和 S75 的 channel-layout 训练稳定完成；S25 仍在第 157 轮发散，而且最佳权重略差于未裁剪 channel-layout。因此裁剪不是统一保证，应逐场景用验证集判断。任何裁剪结果都要与发布配置分表报告。

### 6.6 门控内存布局 `--gate-layout`

可选值：

- `released`：作者发布的 HybridGate reshape 行为，严格复现使用它。
- `channel`：先转成 channel-last 再执行线性层的诊断实现。

这不是普通超参数，而是网络实现变体。当前 channel-layout 降低了 S25、S75 的误差，并显著减少 S75 的周期尖峰，但没有证据证明作者实现错误。论文复现和该变体必须分开呈现。

### 6.7 每轮重新设种子 `--released-epoch-reseed`

作用：复现作者发布训练循环的特殊随机行为。

- 开启：更接近发布代码；当前严格复现采用。
- 关闭：更常规的随机训练顺序；当前小子集 S25 在第 5 轮发散。

它应作为实现对照开关，不应在测试集上挑选。

### 6.8 缺失率和缺失通道 `--data`

通过选择 S25、S50、S75 数据目录改变缺失场景。论文对照必须使用补充材料给出的固定通道组合，不能每次随机选缺失通道。如果研究泛化能力，可以另建“随机通道遮蔽”实验，但要与论文固定场景分开。

### 6.9 数据划分 seed `prepare_hardanger.py --seed`

作用：决定 119 个窗口进入 train/val/test 的清单。

当前固定为 42。不要重复改变划分后选择测试表现最好的划分。更严谨的做法是先锁定一个划分做论文对照，再用多个预注册划分报告敏感性。

### 6.10 窗长、步长、初始裁剪和数据月份

当前脚本中窗长 1024、步长 1024、初始裁剪 867 写在预处理代码中，未暴露为命令行参数。论文对照不应修改。真正最值得扩展的是数据月份，因为当前只有 119 个窗口；扩充月份时应保持同样处理协议，并记录每个压缩包哈希。

### 6.11 损失权重

发布 `JointTimeFreqLoss` 为：

```text
L = spatial_weight * MSE(pred, target)
  + freq_weight * (MSE(real FFT) + MSE(imag FFT)) / 2
```

发布代码默认 `spatial_weight=1`、`freq_weight=1`，FFT 使用二维 `rfft2(norm='ortho')`。论文配置记录频域权重为 0.5。当前适配器保持发布实现，尚未把权重暴露成 CLI 参数。

如果研究损失权重，建议先增加 `--spatial-weight` 和 `--freq-weight`，再固定其他条件比较 `freq_weight = 0, 0.25, 0.5, 1.0`。`0` 是纯时域消融；`0.5` 对应正文记录；`1.0` 对应发布代码。选择依据应同时包括验证集波形误差和频谱误差。

### 6.12 Adam 参数

当前适配器使用 PyTorch Adam 默认 `betas=(0.9, 0.999)`，与发布脚本的 `b1=0.9, b2=0.999` 一致。它们没有暴露成适配器 CLI。只有在学习率、数据量和梯度裁剪仍不能解决稳定性时，才建议测试 beta；优先保持论文默认。

### 6.13 网络规模参数

Hardanger 网络固定为 16 个传感器、32 个基础特征通道、96 维 patch embedding、6 个 TFGTBlock，输入长度 1024，参数量约 1,805,741。修改 block 数、通道数、patch 或 attention 配置需要编辑作者网络源码并从头训练，属于模型消融或改进，不属于论文复现。应在数据范围确认后再做。

## 7. 推荐调参顺序

为避免一次改变多个因素，建议按以下顺序运行：

1. 锁定数据清单、学习率 0.01、batch 16、epoch 200、released gate、每轮 seed 重置，建立严格基线。
2. 保持其余参数不变，只加入 `clip-grad=1.0`，判断是否解决非有限损失。
3. 若仍不稳定，单独比较学习率 0.003 和 0.001；结果标记为学习率诊断。
4. 在同一稳定训练设置下比较 released/channel gate，作为网络实现消融。
5. 再比较频域损失权重 0、0.5、1.0，检查频谱误差和时域误差的权衡。
6. 数据月份扩充并锁定后，再做五个预定种子和模型规模实验。

每一步只看训练集和验证集。配置确定后只评估一次测试集。所有运行使用新的输出目录，不能覆盖历史结果。

## 8. 当前结果

### 8.1 发布布局、论文环境结果

| 场景 | 状态 | 最佳轮次 | 全局 MAE (g) | 全局 RMSE (g) | 全局 RE |
|---|---|---:|---:|---:|---:|
| S25 | 完成 | 166 | 0.000488835 | 0.001304484 | 0.310132 |
| S50 seed 42 | 第 5 轮发散 | — | — | — | — |
| S50 seed 0 | 第 4 轮发散 | — | — | — | — |
| S75 | 完成 | 199 | 0.001620633 | 0.002465538 | 0.586165 |

S25 缺失位置 MAE/RMSE 为 0.001955339/0.002608967 g；相对零填充下降 50.85%/47.34%。
S75 缺失位置 MAE/RMSE 为 0.002160844/0.002846958 g；相对零填充下降 37.03%/34.47%。

### 8.2 当前诊断性最佳结果

| 场景 | 选用方法 | 全局 MAE (g) | 全局 RMSE (g) | 全局 RE | 缺失位置 MAE/RMSE (g) |
|---|---|---:|---:|---:|---:|
| S25 | released + channel 固定等权集成 | 0.000423655 | 0.001167992 | 0.277682 | 0.001694620 / 0.002335984 |
| S50 | channel + clip 1.0 | 0.000731372 | 0.001416634 | 0.336795 | 0.001462744 / 0.002003423 |
| S75 | released + channel-clip 固定等权集成 | 0.001267971 | 0.001981750 | 0.471148 | 0.001690628 / 0.002288328 |

相对零填充，三种场景的缺失位置 RMSE 分别下降 52.85%、57.05%、47.33%。等权集成直接平均两个预测，没有用测试集拟合权重。

这些结果混合了不同诊断配置，并且是在查看多次试验后选出的稳定方法，适合展示改进方向，不适合直接作为论文复现主表。

## 9. 与论文 Source Data 的定量对比

论文官方 Source Data 给出 TFGFormer 五次独立运行的均值：

| 场景 | 论文 MAE (g) | 论文 RMSE (g) | 论文 RE |
|---|---:|---:|---:|
| S25 | 0.0000990 | 0.0002844 | 0.013829 |
| S50 | 0.0003104 | 0.0006252 | 0.030373 |
| S75 | 0.0007372 | 0.0012350 | 0.0599854 |

把当前诊断性最佳全局结果与论文均值比较：

| 场景 | MAE 倍数 | RMSE 倍数 | RE 倍数 | 判断 |
|---|---:|---:|---:|---|
| S25 | 4.28× | 4.11× | 20.08× | 差距最大，远未达到论文精度 |
| S50 | 2.36× | 2.27× | 11.09× | 相对零填充有效，但仍明显落后 |
| S75 | 1.72× | 1.60× | 7.85× | 三场景中最接近论文，仍未复现 |

因此当前能确认的是：模型确实从已知多通道信号中学到了比零填充更好的重建；当前不能确认论文报告的精度已经复现。

## 10. 应如何进行公平的论文对比

### 10.1 先比较实验协议

逐项确认：数据月份、16 路传感器顺序、Z 分量、单位、867 点裁剪作用域、无效段处理、1024 点无重叠切窗、7:2:1 清单、固定缺失通道、五个种子。协议不同，指标差距不能只归因于模型。

当前已经对齐采样率、方向、窗口长度、步长、无额外滤波/归一化和缺失通道；仍未确认月份范围、通道映射、裁剪作用域、无效序列细节及精确划分。

### 10.2 严格配置和诊断配置分表

- 严格表：released gate、无梯度裁剪、正文学习率 0.01、200 epoch、batch 16。
- 发布代码敏感性表：学习率 0.001、发布损失、发布每轮 seed 行为。
- 改进表：channel gate、gradient clipping、ensemble、损失权重变化。

不能把每个场景挑出的最佳变体混在一张表里称为论文复现。

### 10.3 指标口径一致

论文表格使用完整重建矩阵的全局 MAE、RMSE、RE。本工程 `model_global` 与该口径对齐：所有测试窗口、16 路传感器和 1024 个时间点合并计算。

`model_missing_only` 只计算被遮蔽位置，适合说明模型实际填补能力，但不能直接与论文全局表格相比。观测位置由原值替换，因此全局指标会随缺失率和观测比例变化。

公式为：

```text
MAE  = mean(|Y_hat - Y|)
RMSE = sqrt(mean((Y_hat - Y)^2))
RE   = ||Y_hat - Y||_F / ||Y||_F
```

论文报告五次运行均值和标准差；完整复现也必须使用五个预定种子，报告 mean ± sample std，而不是只报告最好一次。

### 10.4 定量分析

至少报告：

1. 当前值、论文均值、绝对差和倍数。
2. 五次运行均值、标准差和每次原始值。
3. 零填充基线及相对改善率。
4. 全局指标和仅缺失位置指标。
5. R²、SNR 作为补充，不能代替论文三项主指标。

当前 RE 的倍数差距远大于 MAE/RMSE，说明当前误差相对于测试信号能量仍然过高。它可能同时受到单月数据分布、测试窗口能量、传感器映射和瞬态预测错误影响，不能简单解释为某一个超参数问题。

### 10.5 定性分析

固定样本和固定缺失通道，不挑选最好看的样本，比较：

- 时域：相位、振幅、均值偏移、强瞬态和尖峰。
- 频域：主模态频率、谱峰幅值、伪峰和高频噪声。
- 残差：误差是否有偏置或周期结构。
- 热图：错误集中在哪些通道和时间段。

当前 S75 图显示 channel-layout + clip 在大部分 102.4 秒窗口内显著压低发布布局的周期性尖峰，误差热图更暗；约 96–102 秒的强瞬态仍是主要误差。S25 仍有均值偏移和末端瞬态问题。论文强调频域保真，因此仅凭 MAE/RMSE 下降不能宣称方法完全复现。

## 11. 可视化与比较命令

比较相同测试数组上的发布布局和变体：

```bash
.venv/bin/python scripts/plot_run_comparison.py \
  --released results/hardanger_S75_fig4_200ep_torch23_releasedloop \
  --variant results/hardanger_S75_fig4_200ep_torch23_channelgate_clip1 \
  --variant-label "Channel layout + clip" \
  --sample 0 \
  --output results/S75_comparison_new
```

创建两个已完成模型的固定等权集成：

```bash
.venv/bin/python scripts/create_fixed_ensemble.py \
  results/RUN_A results/RUN_B \
  --output results/ENSEMBLE_NEW
```

生成三种缺失率与论文均值汇总：

```bash
.venv/bin/python scripts/plot_scenario_summary.py \
  --s25 results/hardanger_S25_torch23_fixed_ensemble \
  --s50 results/hardanger_S50_fig4_200ep_torch23_channelgate_clip1 \
  --s75 results/hardanger_S75_torch23_fixed_ensemble \
  --paper results/paper_targets_hardanger.json \
  --output results/hardanger_summary_new
```

## 12. 重要路径

### 论文、补充材料和官方基准

- 本地工作目录中的 `paper/article.pdf`、`paper/supplementary.pdf` 和
  `paper/supplementary_data.xlsx` 用于分析；根据本仓库 `RIGHTS.md`，这些文件不在 GitHub 再分发。
- 论文正文：https://doi.org/10.1038/s44172-026-00788-0
- `results/paper_source_hardanger_metrics.json`：从 XLSX 提取的逐次值。
- `results/paper_targets_hardanger.json`：论文五次运行均值与标准差。

### 作者代码与核心网络

- `upstream/TFGFormer-for-SHM/TFGFormer/HardangerBridge/train.py`：作者发布训练脚本；
  GitHub 克隆后先运行 `scripts/restore_dependencies.py` 获取固定上游版本。
- `upstream/TFGFormer-for-SHM/TFGFormer/HardangerBridge/Network/TFGFormer.py`：主网络。
- `upstream/TFGFormer-for-SHM/TFGFormer/HardangerBridge/Utils/TimeFrequencyLoss.py`：联合损失。
- `upstream/TFGFormer-for-SHM/Readme.md`：作者 README。

### 本工程关键脚本

- `scripts/fetch_hardanger.py`：下载并校验官方数据文件。
- `scripts/prepare_hardanger.py`：解析 MAT、通道排序、裁剪、切窗、划分和制造缺失。
- `scripts/check_dataset.py`：验证数据配对、形状、掩码和划分。
- `scripts/smoke_model.py`：模型前向、反向和学习率更新冒烟测试。
- `scripts/train_hardanger.py`：主训练、验证、最佳权重选择、测试和基础绘图。
- `scripts/model_variants.py`：released/channel gate 切换。
- `scripts/evaluate_saved_best.py`：评估发散运行留下的最佳权重。
- `scripts/compare_experiments.py`：在完全相同测试数组上比较多个运行。
- `scripts/create_fixed_ensemble.py`：固定 50/50 预测集成。
- `scripts/plot_run_comparison.py`：波形、频谱、残差、柱状图和误差热图。
- `scripts/plot_scenario_summary.py`：三缺失率与论文结果总览。
- `scripts/read_source_xlsx.py`：无需 pandas 读取官方 XLSX。

### 数据与协议

- `configs/hardanger_protocol.json`：论文设置、发布代码差异和未确认项。
- `data/processed/hardanger_2014_10_fig4/manifest.json`：完整数据来源、映射、划分与窗口清单。
- `data/processed/hardanger_2014_10_fig4/checks.json`：数据验证结果。
- `requirements.txt`：环境版本。

### 主要复现结果

- `results/hardanger_S25_fig4_200ep_torch23_releasedloop/`：S25 发布布局。
- `results/hardanger_S75_fig4_200ep_torch23_releasedloop/`：S75 发布布局。
- `results/hardanger_S25_fig4_200ep_torch23_channelgate/`：S25 channel gate。
- `results/hardanger_S50_fig4_200ep_torch23_channelgate_clip1/`：S50 channel + clip。
- `results/hardanger_S75_fig4_200ep_torch23_channelgate_clip1/`：S75 channel + clip。
- `results/hardanger_S25_torch23_fixed_ensemble/`：S25 等权集成。
- `results/hardanger_S75_torch23_fixed_ensemble/`：S75 等权集成。
- `results/hardanger_selected_methods_summary_v2.json`：当前三场景数值汇总。
- `results/hardanger_selected_methods_summary_v2.png`：当前结果与论文均值总览。
- `results/S25_released_channel_ensemble_visual.png`：S25 详细可视化。
- `results/S75_released_channelclip_ensemble_visual_v2.png`：S75 详细可视化。
- `RESULTS_zh.txt`：按时间记录的实验结果。
- `STATUS.json`：机器可读的最新状态。

## 13. 下一阶段建议

第一优先级是确认或扩大 Hardanger 数据月份范围，同时保持当前预处理和学习率 0.01。随后确认传感器映射和 867 点裁剪作用域，锁定数据清单，再完成 S25/S50/S75 各五个预定种子。只有这些条件满足后，才适合判断与论文表格的差距主要来自训练实现还是数据范围。

在当前 119 个窗口上继续大量搜索学习率、seed 或模型变体容易过拟合这一次划分。当前诊断已经足以说明梯度稳定性、门控布局和周期伪峰之间存在关系；下一步增加数据比继续挑参数更有价值。
