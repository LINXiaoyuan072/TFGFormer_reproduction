# TFGFormer_reproduction — main

## 最新复现更新

完整流程、训练参数、逐项调参方法和论文对比分析见
[REPRODUCTION_GUIDE_zh.md](REPRODUCTION_GUIDE_zh.md)。本次更新增加了 S25/S50/S75
诊断训练、固定等权集成、波形/频谱/残差/误差热图和三场景论文基准对照。
正式复跑保持正文学习率 `0.01`；诊断变体与发布配置在文档和结果中分别标注。
适合直接分享给小组成员的版本见
[GROUP_REPRODUCTION_SUMMARY_zh.md](GROUP_REPRODUCTION_SUMMARY_zh.md)。

这是 Hardanger Bridge 的子集方法验证，不能据此宣称已复现论文表格。
上游代码未随仓库再分发，且尚未确认其许可证。运行 `python scripts/restore_dependencies.py` 可从作者仓库获取固定 commit，
使用前请核对作者许可；论文与补充材料通过 DOI 阅读，不在本仓库复制。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/restore_dependencies.py
python scripts/fetch_hardanger.py MidFreq_2014_10.zip
python scripts/prepare_hardanger.py
python scripts/check_dataset.py
python scripts/smoke_model.py
python scripts/train_hardanger.py --epochs 200 --lr 0.01 --seed 42 --released-epoch-reseed --output results/hardanger_S25_paper_env_seed42
```

requirements.txt 对应论文环境。最新本地记录已验证 Python 3.10.22 / PyTorch 2.3.0+cu121 / cuDNN 8.9.2；
早期实验使用 PyTorch 2.4/Python 3.11。主结果见 STATUS.json 与 results/hardanger_S25_fig4_200ep_torch23_releasedloop。
--released-epoch-reseed 对应作者训练循环每轮重新设种子的行为；不加该参数的结果不能与主实验混同。
S50 存在非有限损失，S25/S75 仍有周期伪峰，尚未复现论文的时频保真度，详见最新结果记录。
CUDA 库复用脚本仅用于原开发机环境重建；新机器应优先按 requirements.txt 安装官方依赖。
训练会拒绝覆盖已有输出目录；请为新实验选择新目录。详见
[NEXT_STEPS_zh.txt](NEXT_STEPS_zh.txt)、[RESULTS_zh.txt](RESULTS_zh.txt) 与
[STATUS.json](STATUS.json)。
公开指标需注明数据月份、通道顺序假设、单位和完整矩阵/缺失位置；部分论文设置仍有待作者确认。
自写适配脚本尚未另行指定开源许可证；公开可见不代表取得上游或本仓库全部内容的再分发许可。
论文：https://doi.org/10.1038/s44172-026-00788-0
数据：https://doi.org/10.21400/5ng8980s
代码归档：https://doi.org/10.5281/zenodo.21979714


源码与权利说明见 [UPSTREAM.json](UPSTREAM.json) 和 [DATA_AND_ENVIRONMENT.md](DATA_AND_ENVIRONMENT.md)。
