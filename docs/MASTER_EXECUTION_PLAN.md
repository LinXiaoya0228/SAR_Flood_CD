# ChangeGuard FYP 总执行计划

版本：1.0  
计划起点：2026 年 10 月 6 日  
Draft Final Report：2027 年 3 月 25 日  
Final Report：2027 年 4 月 9 日  
Project Demonstration：2027 年 4 月 12–16 日  
Oral Presentation：2027 年 5 月 10–12 日  
DR-NTU Full Text：2027 年 5 月 19 日

---

# 一、项目最终定位

## 1.1 暂定题目

**Failure-Aware Test-Time Adaptation for SAR Flood Change Detection: Event-Wise Characterization and Selective Adaptation**

中文：

**面向 SAR 洪水变化检测的失效感知测试时自适应：事件级分析与选择性自适应**

如果后续实验表明 controller 不成立，但 consistency 方法成立，可改为：

**Stable Test-Time Adaptation for SAR Flood Change Detection under Event-Level Distribution Shift**

如果所有自适应方法都不能稳定超过 source-only，可改为：

**When Test-Time Adaptation Fails in SAR Flood Change Detection: An Event-Wise Empirical Study**

因此，本项目不是“只有 ChangeGuard controller 成功才能毕业”。无论最终落在哪条分支，都能形成逻辑完整的 FYP。

---

# 二、已经完成的工作

## 2.1 数据

Kuro Siwo 已完整下载并通过结构审查：

- 43 个洪水事件；
- 31,707 个训练 manifest 样本；
- 67,490 个测试 manifest 样本；
- 随机读取 100 个样本，没有不可读文件；
- 每个样本包含核心 Sentinel-1、标签、DEM 等组件；
- 数据总量约 782 GB。

标签语义：

- 0：No Water；
- 1：Permanent Water；
- 2：Flood；
- 3：Invalid。

主要任务是三分类预测，但主要报告 class 2 Flood 的 Precision、Recall、F1 和 IoU。

## 2.2 数据划分

正式事件划分已经固定：

- Source training：27 个事件；
- TTA validation：6 个有效 validation events；
- Final test：10 个事件，仍然封存。

TTA validation events：

```text
279, 437, 520, 559, 1111003, 1111008
```

封存 test events：

```text
205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013
```

未经最终方法冻结和导师批准，不得读取 test event 的图像、标签、预测或指标。

## 2.3 官方 checkpoint 的问题

目前已经确认：

- checkpoint 第一层权重为 `(32, 3, 3, 3)`；
- 因此每个时间点需要 3 个输入通道；
- 当前官方配置主要写 VV、VH；
- 第三通道没有得到官方材料的完全确认；
- GitHub 未合并 PR 推测其为 slope，但这只能当作线索，不能当作证明；
- checkpoint 可以 strict load，模型可以正常 forward；
- 但 checkpoint 的输入 provenance 仍然不够完整。

所以：

> 官方 checkpoint 保留为 secondary reference，不作为论文唯一主模型。

## 2.4 两周 pilot 的主要发现

目前 pilot 是基于官方 checkpoint 和暂定预处理完成的，因此结论是“checkpoint-conditional”，不能直接当作最终论文结果。

观察到：

- Source-only event-macro Flood F1 约为 0.5646；
- Source-only pooled Flood F1 约为 0.7020；
- Tent 和 AdaBN 的 event-macro F1 约为 0.543–0.545；
- 某些事件严重下降；
- 某些事件，例如 Event 559，可能从适应中受益；
- 504/537 个更新批次在预定义标准下被标记为 harmful；
- 初版 change-specific risk score AUROC 约为 0.5102；
- pooled BN drift 看起来较强，但 event-balanced correlation 接近零；
- oracle event selector 明显好于 source-only；
- 当前 label-free selector 基本退化成 reject-all/source-only。

这些结果说明：

1. SAR 洪水检测中的 TTA 确实可能发生严重失效；
2. 事件之间差异很大；
3. pooled 指标可能掩盖 event-wise failure；
4. 简单风险分数和 controller 目前尚未成功；
5. 需要在完全可控的自训练模型上重新验证这些结论。

---

# 三、必须纠正的几个说法

## 3.1 “有四张 3090，所以可以同时占满四张”

不完全正确。

服务器有四张 GPU，不等于这四张卡归一个用户独占。这是公用服务器。每次启动任务前必须检查：

- 哪些 GPU 正在使用；
- 使用率；
- 显存占用；
- 进程是否属于其他用户。

默认先使用一张空闲 GPU。只有在其余 GPU 明确空闲且符合实验室公平使用政策时，才能增加到两张或更多。

绝不能：

- 杀掉其他用户进程；
- 在不知道占用者的情况下抢卡；
- 默认连续数天占满四张卡。

## 3.2 “四张 3090 相当于一张 96 GB GPU”

错误。

每张 3090 是独立的 24 GB 显存。四个独立实验可以分别使用四张卡，但单个普通训练任务仍然只能使用其中一张卡的 24 GB，除非专门实现 DDP 或模型并行。

当前 SNUNet 不需要多卡训练。最合理的方式是：

- 每张卡运行一个独立 seed/config；
- 不做复杂多卡同步训练。

## 3.3 “官方 SNUNet 只训练 30 epochs”

不能这样写。

Kuro Siwo 官方材料中 change-detection models 的主要训练设置是约 150 epochs。后续自建 source model 应优先复现官方训练策略，然后根据收敛曲线决定是否缩短。

30 epochs 可以用于：

- smoke test；
- pipeline 调试；
- 初步比较；

但不能直接当作正式训练标准。

## 3.4 “第三通道就是 slope”

尚未证明。

当前合理做法是：

- 2-channel `[VV,VH]`：主候选模型；
- 3-channel `[VV,VH,slope]`：预声明 ablation；
- 官方 checkpoint + slope：仅作为 reconstruction/reference。

不能把 slope 猜测写成官方事实。

## 3.5 “一定能发 TGRS、一定能拿 A/A+”

不能保证。

当前项目：

- 完成高质量 FYP：可行性高；
- 写出完整论文初稿：可行；
- 挂 arXiv：取决于导师同意和论文成熟度；
- 正式投稿：可行；
- 被 TGRS 接收：目前无法保证。

TGRS 需要比 FYP 更完整的实验，包括多模型、多 shift、统计检验、消融和可复现性。不能以一次两周 pilot 作为录用依据。

---

# 四、最终研究策略：双线但有主次

## Track A：官方模型参考线

目标不是依赖它完成论文，而是回答：

- 官方 checkpoint 能达到什么结果？
- slope reconstruction 是否合理？
- 官方结果与自训练模型相差多少？

只做有限工作：

1. 保留当前官方 checkpoint 结果；
2. 使用最合理的 `[VV,VH,slope]` 预处理再验证一次；
3. 保存输出分布、event-wise 指标和 provenance；
4. 完成后冻结，不再围绕黑盒反复调参。

## Track B：自主训练的受控主线

这是正式论文主线。

训练两个 SNUNet source variants：

### Model A：主候选

```text
每个时间点：[VV, VH]
输入对：(pre2, post)
输出：No Water / Permanent Water / Flood
```

### Model B：通道消融

```text
每个时间点：[VV, VH, slope]
输入对：(pre2, post)
输出：No Water / Permanent Water / Flood
```

Model A 和 Model B 必须保持相同：

- 训练事件；
- epoch 数；
- loss；
- optimizer；
- augmentation；
- batch size；
- seeds；
- model capacity；
- evaluation protocol。

模型选择不能使用 TTA validation 或 test labels。

---

# 五、Source model 的内部划分

27 个 source-training events 需要再划分：

- 22 个 source-inner-train events；
- 5 个 source-inner-validation events。

用途：

- 选择 epoch；
- 选择 loss；
- 比较 2-channel 与 slope；
- 检查 overfitting。

TTA validation 的 6 个事件不能用于选择 source model 训练配方。

流程：

1. 在 22/5 划分上完成配方选择；
2. 记录最佳 epoch 或收敛 epoch；
3. 冻结配方；
4. 使用全部 27 个 source events 重训最终 source model；
5. 最终重训使用固定 epoch，不能再根据 TTA validation 挑 checkpoint。

---

# 六、主实验结构

## 6.1 Source baselines

必须完成：

1. Controlled SNUNet `[VV,VH]`；
2. Controlled SNUNet `[VV,VH,slope]`；
3. Official checkpoint reference；
4. 至少 3 个 random seeds。

主要指标：

- Flood Precision；
- Flood Recall；
- Flood F1；
- Flood IoU；
- macro event F1；
- pooled F1；
- worst-event F1；
- 每个事件的预测洪水面积；
- 标准差和置信区间。

主模型选择优先依据：

1. event-macro F1；
2. worst-event F1；
3. 多 seed 稳定性；
4. provenance 和复现清晰度；
5. 不能只看 pooled F1。

## 6.2 TTA baselines

优先级 P0，必须完成：

- Source-only；
- AdaBN；
- Tent。

优先级 P1，时间允许应完成：

- EATA；
- CoTTA；
- SAR。

优先级 P2，仅在前面稳定后考虑：

- CIM-TTDA 或其他 change-detection-specific TTA。

所有 TTA 对比必须使用相同：

- source checkpoint；
- validation events；
- batch size；
- stream order；
- seeds；
- preprocessing；
- prediction timing；
- reset protocol。

## 6.3 新方法最小版本

先实现三个组件，不直接做复杂 controller：

### Component 1：Confidence masking

只让高可信或稳定像素贡献无监督适应 loss，避免背景和噪声主导更新。

### Component 2：Source anchor

适应模型不能无限偏离冻结 source model。可以约束：

- logits；
- soft probabilities；
- selected feature；
- source-predicted flood regions。

### Component 3：SAR-aware consistency

使用弱增强和强增强，要求模型预测一致。

增强应符合 SAR 场景：

- 轻微 speckle/noise 模拟；
- 合理的 intensity scaling；
- 小范围 spatial shift；
- horizontal/vertical flip；
- 不破坏标签语义的裁剪。

不要直接照搬 RGB color jitter。

后续可选：

- teacher-student EMA；
- class-balanced pseudo labels；
- permanent-water-aware constraint；
- temporal null-pair probe；
- selective rollback/controller。

只有前面三个组件证明有改善，才继续加入 controller。

---

# 七、实验成败判断

## Gate S：Source model 合格

满足以下条件才进入最终 TTA：

- 完整跑完全部 source-inner train/val；
- 没有样本遗漏；
- 三分类输出正常；
- 不是 all-background；
- 多 seed 结果可重复；
- event-macro F1 达到合理水平；
- `[VV,VH]` 和 `[VV,VH,slope]` 都完成公平比较；
- source checkpoint、config、split、代码 commit 都可追踪。

如果自训练 source 明显失败，先修 source，不做 TTA。

## Gate A2：受控模型是否存在 TTA failure

在自主训练 source model 上检查：

- Tent/AdaBN 是否至少在一个事件明显伤害 F1；
- 是否也有事件从适应中受益；
- failure 是否在多个 seed/order 中出现；
- failure 是否不是数据读取或标签映射错误。

如果 harmful adaptation 不再存在：

> 停止 ChangeGuard controller，转向“checkpoint/preprocessing sensitivity”或普通稳定 TTA。

如果所有事件都只受害、几乎没有任何事件受益：

> 不值得做 controller；转向 SAR TTA failure characterization。

如果既有受益事件又有受害事件：

> 选择性适应/ChangeGuard 才有研究空间。

## Gate M：新方法是否值得继续

在 validation 上至少满足：

- macro F1 比 source-only 提高约 1–2 个百分点，或；
- worst-event F1 明显改善，且平均性能不下降；
- 至少两个事件受益；
- 不是仅靠 Event 559；
- 不能只是 reject-all；
- 多 seed/order 稳定。

如果达不到，不继续堆复杂模块，保留 failure-analysis 论文方向。

## Gate C：Controller 是否可做

只有当未保护方法同时存在：

- 明显 beneficial updates；
- 明显 harmful updates；

才实现 controller。

Controller 必须：

- 不访问标签；
- 使用固定风险公式；
- 阈值用 leave-one-event-out 选择；
- 不能拒绝所有更新；
- 能保留一部分有益适应；
- 能减少 worst-event degradation。

## Final Gate：解封 test

必须同时满足：

- source model 和 preprocessing 冻结；
- 方法结构冻结；
- 所有阈值冻结；
- validation 结果完成；
- 主要消融完成；
- 代码和 configs 已 commit；
- supervisor 同意；
- test 执行脚本事先审查；
- 明确只运行一次主实验。

---

# 八、GPU 与环境使用标准

## 8.1 什么时候用 base

`base` 只做：

- Git；
- 文件查看；
- tmux；
- GPU 状态查看；
- 普通系统操作。

## 8.2 什么时候用 vfm_env

所有以下操作都使用 `vfm_env`：

- PyTorch；
- dataset loader；
- model forward；
- training；
- evaluation；
- TTA；
- metrics 和 plots。

每次运行模型前：

```bash
conda activate vfm_env
```

## 8.3 启动任务前检查 GPU

```bash
nvidia-smi
```

确认：

- 哪张卡显存基本空闲；
- GPU utilization 较低；
- 没有其他用户的重要任务。

如需检查进程：

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

不要根据一个瞬间的低利用率就判断 GPU 空闲，因为模型可能正在数据加载。

## 8.4 分配单个 GPU

如果选物理 GPU 2：

```bash
CUDA_VISIBLE_DEVICES=2 python tools/train_controlled_source.py --config configs/source/snunet_vvvh_seed0.json
```

Python 程序内部仍应使用：

```python
device = torch.device("cuda:0")
```

因为对这个进程来说，物理 GPU 2 已被重新编号为可见的 GPU 0。

## 8.5 多个独立实验

只有确认多张卡空闲时才启动，例如：

```text
GPU 0：VV/VH seed 0
GPU 1：VV/VH seed 1
GPU 2：VV/VH/slope seed 0
GPU 3：暂时保留或运行 seed 1
```

默认先启动一到两个任务，不默认占满四张卡。

## 8.6 tmux

长任务使用 tmux。正常 terminal 也能训练，但 SSH 或 Wi-Fi 断线可能导致任务结束。

创建：

```bash
tmux new -s cg_vvvh_s0
```

退出而不停止任务：

```text
Ctrl-b，然后按 d
```

查看：

```bash
tmux ls
```

重新进入：

```bash
tmux attach -t cg_vvvh_s0
```

训练脚本必须支持：

- checkpoint resume；
- 每个 epoch 保存日志；
- 定期保存 last checkpoint；
- 最佳 checkpoint；
- interruption recovery。

tmux 不能替代 checkpoint。服务器重启后 tmux 也会消失。

## 8.7 公用服务器规则

不得：

- 在 prompt、代码或 Git 中保存服务器密码；
- 杀掉不属于自己的进程；
- 默认占满所有 GPU；
- 把 dataset 和 checkpoint 提交到 Git；
- 假设服务器会永久保存文件。

MLDA 服务器不保证备份，所以：

- 代码至少每周备份一次；
- 关键 CSV、JSON、plots、论文和 config 备份；
- 最佳 checkpoint 至少保存两份；
- 782 GB 原始数据可以不完整备份，但下载清单和 checksum 必须保存。

---

# 九、统一实验记录标准

每次正式运行必须有唯一 run ID：

```text
YYYYMMDD_model_channels_method_seed_order
```

例如：

```text
20261118_snunet_vvvh_source_seed0_fixed
20261202_snunet_vvvh_tent_seed1_order2
```

每次运行保存：

```text
config.json
git_commit.txt
command.txt
environment.txt
split_hash.txt
checkpoint_hash.txt
stdout.log
metrics_per_event.csv
metrics_pooled.json
run_manifest.json
completion.json
```

`run_manifest.json` 至少记录：

- start/end time；
- host；
- GPU；
- seed；
- event IDs；
- 样本数；
- checkpoint hash；
- config hash；
- code commit；
- success/failure；
- exception；
- 是否访问 GT；
- 是否访问 test。

禁止手工修改 generated CSV。若代码有 bug：

1. 保留原始结果；
2. 建新目录；
3. 标记 correction；
4. 记录旧、新 hash；
5. 说明变化原因。

---

# 十、每周工作模式

每周只有 2–3 天可以工作时，采用以下结构。

## 工作日 A：设计与代码

- 阅读上周日志；
- 写或审查代码；
- 跑 unit tests；
- 做小规模 smoke test；
- 固定 config。

## 工作日 B：启动正式实验

- 检查 GPU；
- 启动 tmux；
- 记录 run ID、命令和 GPU；
- 检查前几个 batches；
- 离开服务器，让任务后台运行。

## 工作日 C：结果与写作

- 检查 completeness；
- 生成 event-wise 表和图；
- 判断是否达到该周 Gate；
- 更新 experiment log 和论文；
- commit 代码和文档；
- 备份关键结果。

不要等到三月才开始写报告。每周都要更新论文对应部分。

---

# 十一、逐周计划

## Week 0：2026-10-06 至 2026-10-11

目标：冻结自主训练 protocol。

任务：

1. 备份当前 ChangeGuard 仓库；
2. 确认当前 Git clean/dirty 状态；
3. 创建 22/5 source-inner event split；
4. 计算并保存 split hash；
5. 写 source training protocol；
6. 明确 2-channel 和 slope 两个配置；
7. 建立 interim report 大纲；
8. 检查服务器 GPU 使用规范。

AI 可以：

- 检查数据 loader；
- 生成 split 脚本；
- 创建 configs；
- 写 unit tests；
- 起草 protocol。

你必须：

- 审核 event IDs；
- 决定并批准 split；
- 检查 GPU；
- 和导师确认研究方向。

完成标准：

- source-inner split 被固定；
- protocol 文档存在；
- test events 未访问；
- configs 可被加载；
- Git 有一个清晰 checkpoint commit。

## Week 1：2026-10-12 至 2026-10-18

目标：完成自主 source training pipeline。

任务：

1. 实现 `[VV,VH]`；
2. 实现 `[VV,VH,slope]`；
3. 验证 slope 与 SAR patch 几何对应；
4. 实现 CE+Dice、invalid mask；
5. 支持 checkpoint resume；
6. 支持固定 epoch 与 best-inner-val checkpoint；
7. 每个配置跑 10–50 个 batch smoke test；
8. 检查 loss 是否下降。

完成标准：

- 两种输入都能 forward/backward；
- 无 NaN/Inf；
- label 3 不参与 loss；
- checkpoint 能保存和恢复；
- smoke run 有 manifest；
- unit tests 全部通过。

不要开始正式四卡并行，直到 smoke test 通过。

## Week 2：2026-10-19 至 2026-10-25

目标：跑第一轮 source seed 0。

实验：

- Model A `[VV,VH]` seed 0；
- Model B `[VV,VH,slope]` seed 0；
- 按官方训练设置启动正式训练；
- 记录训练/inner-val curves。

并行策略：

- 最多用两张明确空闲 GPU；
- 一个 GPU 跑一个模型；
- 如果只有一张空闲卡，顺序运行。

完成标准：

- 两个 seed 0 正式 run 完整结束；
- 样本覆盖完整；
- 有 train/val loss 和 F1 curves；
- 没有中断未记录；
- checkpoint hashes 已保存。

## Week 3：2026-10-26 至 2026-11-01

目标：重复性与 preliminary source comparison。

实验：

- `[VV,VH]` seed 1；
- `[VV,VH,slope]` seed 1；
- 如果时间允许跑 seed 2；
- 官方 checkpoint slope reconstruction 仅做一次 inference reference。

分析：

- event-macro F1；
- pooled F1；
- worst-event；
- seed variation；
- slope 是否稳定改善。

完成标准：

- 至少两个 seed；
- 有公平的 A/B 表格；
- 官方 checkpoint 被明确标为 reference；
- 没有用 TTA validation 选择 source 配方。

## Week 4：2026-11-02 至 2026-11-08

目标：准备 Interim Report 和视频。

内容：

1. 研究问题；
2. Kuro Siwo 数据语义；
3. 数据完整性；
4. 官方 checkpoint provenance 问题；
5. 两周 pilot 结果；
6. controlled source training 设计；
7. seed 0/1 preliminary results；
8. 后续计划和风险。

完成标准：

- Interim Report 初稿；
- 所有图有标题、坐标和说明；
- 视频 slide 完成；
- 导师至少看过一次；
- 不夸大 preliminary 结果。

## Week 5：2026-11-09 至 2026-11-15

重要截止：2026-11-10 Interim Report & Video。

目标：

- 按时提交；
- 修复导师提出的紧急问题；
- 冻结 source model recipe。

如果 2-channel 与 slope 差异小：

- 主模型选 provenance 更清楚的 `[VV,VH]`；
- slope 保留为 ablation。

如果 slope 明显且稳定更好：

- 可把 slope 作为主模型；
- 但必须清楚说明其生成方式和物理意义。

完成标准：

- Interim 正式提交；
- source recipe、epoch、loss、augmentation 固定；
- Model A/B 选择规则写入文档。

## Week 6：2026-11-16 至 2026-11-22

目标：在全部 27 个 source events 上训练最终 source model。

实验：

- 主模型 seeds 0/1/2；
- 固定 epoch；
- 不使用 TTA validation 选 checkpoint；
- slope variant 至少保留一个或三个 seed，取决于 Week 5 结果。

完成标准：

- 三个 primary source checkpoints；
- 每个有完整 provenance；
- checkpoint 可重新加载；
- hashes 已保存；
- 核心结果已备份。

## Week 7：2026-11-23 至 2026-11-29

目标：冻结 controlled source-only validation baseline。

在六个 TTA validation events 上运行，但暂不适应。

输出：

- 每事件 F1、IoU、Recall、Precision；
- pooled 与 event-macro；
- worst-event；
- predicted flood area；
- error visualizations；
- 三个 seed 的平均和标准差。

完成标准：

- 没有 all-background；
- 没有读取 test；
- 结果完整；
- 数据读取和标签映射通过审查；
- Gate S 通过。

若 Gate S 不通过，后面计划顺延，优先修 source。

## Week 8：2026-11-30 至 2026-12-06

目标：在 controlled source 上复现核心 TTA failure。

实验：

- Source-only；
- AdaBN；
- Tent；
- 三个 seeds；
- 至少两个 stream orders；
- reset-per-event 和预声明 continual protocol。

完成标准：

- 每次 run 完整；
- event-wise degradation 被量化；
- 明确哪些事件受益、哪些事件受害；
- 旧 pilot 与新结果作比较；
- 通过或否决 Gate A2。

## Week 9：2026-12-07 至 2026-12-13

目标：补充现代 TTA baselines。

优先顺序：

1. EATA；
2. CoTTA；
3. SAR。

先做单事件 smoke test，再做六事件 validation。

完成标准：

- 至少完成两个现代 baseline；
- 每个 baseline 使用统一协议；
- 不允许针对单一事件单独调参；
- 所有失败 run 也保留。

## Week 10：2026-12-14 至 2026-12-20

目标：完成 baseline matrix 和路线决策。

回答：

- controlled model 上 harmful adaptation 是否存在？
- 是否有事件受益？
- strongest generic TTA 是谁？
- failure 是否跨 seed/order 稳定？
- 是否值得做 selective adaptation？

同时做 IGARSS go/no-go。

只有以下条件同时满足才考虑 IGARSS：

- 核心结果稳定；
- 论文主张已经明确；
- 12 月 20 日前有完整主表；
- 不会拖延 FYP 主线。

否则跳过 IGARSS，不强行投稿。

## Week 11：2026-12-21 至 2026-12-27

目标：实现 SAR-aware consistency 最小版本。

实现：

- weak augmentation；
- strong augmentation；
- confidence mask；
- consistency loss；
- source-only anchor 暂不开启。

实验：

- 一个 seed；
- 六个 validation events；
- 固定参数小网格，不能针对每个事件调。

完成标准：

- loss 正常；
- 不是全背景；
- 至少不比 Tent 更差；
- runtime、memory 被记录。

## Week 12：2026-12-28 至 2027-01-03

目标：完成 confidence 和 augmentation ablation。

比较：

- 无 mask；
- confidence mask；
- 不同 confidence threshold；
- RGB-style augmentation 禁用；
- SAR-aware augmentation；
- weak-only 与 weak-strong。

完成标准：

- 阈值在统一 validation protocol 下选择；
- 保留全部结果；
- 明确哪个组件有效；
- 不手工挑选有利事件。

## Week 13：2027-01-04 至 2027-01-10

目标：加入 source anchor。

比较：

- consistency only；
- source logit anchor；
- source probability anchor；
- selected feature anchor。

先用简单固定权重，不训练复杂 risk network。

完成标准：

- 至少一个 anchor 能降低最大性能下降；
- 不是单纯关闭适应；
- source/adapted drift 被量化。

## Week 14：2027-01-11 至 2027-01-17

目标：class imbalance 与 permanent water 分析。

实验：

- class-balanced confidence selection；
- flood-aware sampling；
- permanent-water protection；
- source-predicted flood-region preservation。

分析：

- low-flood event；
- high-flood event；
- permanent-water-heavy event；
- class imbalance 与 degradation 的关联。

完成标准：

- 因果结论保持谨慎；
- 不把相关性写成因果；
- 至少得到一组可解释 ablation。

## Week 15：2027-01-18 至 2027-01-24

目标：temporal semantics。

实验：

- real `(pre2,post)`；
- identical `(pre2,pre2)`；
- identical `(post,post)`；
- natural stability `(pre1,pre2)` 只作辅助。

注意：

- identical pair 应没有时间变化；
- 但 Permanent Water 不要求为零；
- `(pre1,pre2)` 不能赋全零 GT；
- null probe 只产生无标签风险证据。

完成标准：

- null-pair 行为得到可视化；
- 不使用伪造的全零 natural-pair label；
- 明确它是否能提供有用风险提示。

## Week 16：2027-01-25 至 2027-01-31

目标：组合最小 ChangeGuard。

只组合已经在独立实验中有效的组件，例如：

```text
SAR consistency
+ confidence mask
+ source anchor
```

如果 null-pair 没有预测力，不加入主方法。

完成标准：

- 主方法公式冻结；
- 所有 hyperparameters 有来源；
- 主方法不是复杂黑盒；
- 有完整 config 和方法图。

## Week 17：2027-02-01 至 2027-02-07

目标：判断 controller 是否有必要。

如果有受益和有害更新并存：

- 实现 accept/reduce/reject；
- 使用 last-safe state；
- 保留 immutable source。

如果几乎所有适应都受害：

- 不做 controller；
- 把 source-only 或 anchored consistency 作为主方法。

完成标准：

- controller 行动全部有日志；
- 阈值不使用同一 held-out event；
- reject-all 被明确视为失败。

## Week 18：2027-02-08 至 2027-02-14

目标：leave-one-event-out controller evaluation。

六轮：

- 五个 validation events 定阈值；
- 一个 held-out event 测试；
- 对六个事件轮流执行。

报告：

- accepted/reduced/rejected；
- false rejection；
- beneficial updates preserved；
- harmful updates prevented；
- worst-event F1。

完成标准：

- 阈值能够跨事件迁移；
- 不是只对 Event 559 有效；
- 不访问 test。

## Week 19：2027-02-15 至 2027-02-21

目标：三 seed、多 order 主实验。

比较：

- Source-only；
- best generic TTA；
- consistency；
- anchored consistency；
- ChangeGuard/controller，如适用。

完成标准：

- 三 seeds；
- 多 stream orders；
- mean ± std；
- bootstrap CI；
- event-macro 和 pooled 同时报告；
- 不隐藏失败 seed。

## Week 20：2027-02-22 至 2027-02-28

目标：最终 validation 消融和方法冻结。

消融：

- 去掉 confidence mask；
- 去掉 anchor；
- 去掉 consistency；
- 去掉 null probe；
- 去掉 controller；
- 替换 generic protection。

可选增强：

- 第二个轻量 backbone；
- 一个预声明 synthetic shift；
- runtime/memory 分析。

完成标准：

- final config 冻结；
- final code commit；
- test command 预先生成；
- 导师批准解封 test。

## Week 21：2027-03-01 至 2027-03-07

目标：一次性 final test evaluation。

只执行预先冻结的：

- Source-only；
- strongest generic baseline；
- proposed method；
- 必要 ablation，不临时新增调参版本。

规则：

- test 不用于调阈值；
- 看到结果后不修改方法再重跑；
- 如果发现代码 bug，保留原始结果，并书面记录 correction；
- 不运行几十个 test variants。

完成标准：

- 10 个 test events 全部完成；
- event-wise、macro、pooled 指标完整；
- completeness 和 manifests 通过；
- 主表自动生成。

## Week 22：2027-03-08 至 2027-03-14

目标：最终分析。

输出：

- 主结果表；
- event-wise table；
- worst-event analysis；
- ablation；
- qualitative predictions；
- failure cases；
- accepted/rejected updates；
- runtime/memory；
- limitations；
- statistical uncertainty。

完成标准：

- 所有论文 figures 基本完成；
- 结果可从 CSV 自动重建；
- 主张与证据一一对应。

## Week 23：2027-03-15 至 2027-03-21

目标：Draft Final Report 完整初稿。

章节：

1. Introduction；
2. Related Work；
3. Dataset and Task；
4. Pilot and Motivation；
5. Method；
6. Experimental Protocol；
7. Results；
8. Ablation；
9. Failure Analysis；
10. Limitations；
11. Conclusion；
12. Research Integrity and Reproducibility。

完成标准：

- 不留空章节；
- 所有表和图进入正文；
- 导师收到完整 draft，而不是零散片段。

## Week 24：2027-03-22 至 2027-03-28

重要截止：2027-03-25 Draft Final Report。

任务：

- 3 月 22–24 日完成语言和结构审查；
- 3 月 25 日前提交；
- 记录 supervisor feedback；
- 不再新增高风险实验。

完成标准：

- Draft 按时提交；
- 引用和 appendix 完整；
- 代码及结果已备份。

## Week 25：2027-03-29 至 2027-04-04

目标：根据反馈修改 Final Report。

只允许：

- 补充必要分析；
- 修正表达；
- 修复已确认错误；
- 加入少量低风险可视化。

避免：

- 重构整个方法；
- 临时解封 test 调参；
- 新增大规模训练。

完成标准：

- supervisor feedback 全部处理或解释；
- final report 接近冻结。

## Week 26：2027-04-05 至 2027-04-11

重要截止：2027-04-09 Final Report。

任务：

- final proofreading；
- 检查表格数字；
- 检查引用；
- 检查 appendix；
- 检查匿名或署名要求；
- 最终提交。

完成标准：

- 4 月 9 日按时提交；
- PDF 已本地备份；
- 代码、模型、结果归档。

## Week 27：2027-04-12 至 2027-04-18

Project Demonstration：4 月 12–16 日。

准备：

- 数据样本展示；
- source-only 与 TTA 对比；
- harmful event；
- protected method；
- event-wise dashboard；
- 失败案例；
- 可复现实验说明。

演示不能依赖实时长训练。准备好静态结果作为断网备份。

## Week 28–29：2027-04-19 至 2027-05-02

目标：论文版本和 oral presentation。

任务：

- 将 FYP 报告压缩成论文结构；
- 根据 supervisor 意见决定 arXiv/TGRS/会议；
- 不能自动上传 arXiv；
- 如投期刊，核对 venue 的 preprint 和扩展政策。

完成标准：

- 论文初稿；
- 口头报告 slides；
- 回答常见质疑的 Q&A 文档。

## Week 30：2027-05-03 至 2027-05-09

目标：oral rehearsal。

至少做三次完整演练：

- 第一次检查时间；
- 第二次让非本领域同学提问；
- 第三次模拟 examiner。

必须能回答：

- 为什么是 SAR？
- 为什么 pooled 指标不够？
- 为什么不用 official checkpoint 作唯一模型？
- 为什么 TTA 会失败？
- 方法是否只是拒绝更新？
- test 是否被用于调参？
- 物理先验体现在哪里？
- limitation 是什么？

## Week 31：2027-05-10 至 2027-05-16

Oral Presentation：5 月 10–12 日。

报告后立即记录 examiner feedback，为 DR-NTU 版本修订。

## Week 32：2027-05-17 至 2027-05-19

重要截止：2027-05-19 DR-NTU Full Text。

任务：

- 完成最终格式；
- 检查权限和版权；
- 提交 full text；
- 保存 submission confirmation。

---

# 十二、AI / Antigravity 可以做什么

可以交给 AI：

- 阅读代码和文档；
- 实现配置驱动的数据 loader；
- 写训练、evaluation、TTA 脚本；
- 写 unit tests；
- 创建 run manifest；
- 生成 plots 和 LaTeX tables；
- 做 Git diff 审查；
- 检查 test-event ID 是否泄漏；
- 检查 label 是否进入 adaptation；
- 起草报告和论文；
- 将结果整理成 event-wise 表；
- 检查数字在 CSV、文档、表格中是否一致；
- 生成启动命令；
- 帮助恢复中断任务；
- 检查 checkpoint/config hash；
- 起草给导师的问题或进度报告。

每次给 Antigravity 的任务都必须包含：

```text
Before modifying anything, report the current Git branch, commit and dirty state.

Do not access the sealed test events unless this task explicitly states that the final method has been frozen and test evaluation has been approved.

Do not use GT labels in test-time adaptation, risk generation, controller decisions or threshold selection.

Do not delete failed runs, overwrite generated results, manually edit result CSVs, or change the event split.

After the task, report:
1. every file read;
2. every file changed;
3. every command executed;
4. all event IDs accessed;
5. whether GT was accessed and for what purpose;
6. confirmation of no test leakage;
7. all protocol deviations;
8. Git diff and Git status.

Do not commit automatically.
```

---

# 十三、不能完全交给 AI 的工作

必须由你决定或确认：

- 是否占用某张公用 GPU；
- 是否停止某个进程；
- 是否解封 test；
- 是否修改研究问题；
- 是否接受 post-hoc 参数变化；
- 是否把某个结果当作论文主结论；
- 是否提交报告；
- 是否上传 arXiv；
- 是否联系作者；
- 是否提交会议或期刊；
- 是否 commit AI 大规模修改；
- 是否把外部代码许可证允许的内容纳入项目。

AI 不得：

- 编造实验结果；
- 为了提高分数修改 GT；
- 删除不利事件；
- 只报告最佳 seed；
- 看到 test 后继续调参；
- 将 pooled improvement 伪装成 event-wise improvement；
- 在没有证据时声称 slope 是官方第三通道；
- 自动杀掉服务器进程；
- 自动提交或发布论文。

---

# 十四、项目的三档最终成果

## 档位 A：高质量 FYP

最低要求：

- 可控 source model；
- 数据与 split 完整；
- source/AdaBN/Tent 对比；
- event-wise failure analysis；
- 一个合理的稳定化方法或清晰的负面结论；
- 完整可复现实验；
- test 无泄漏；
- 高质量报告与演示。

这是现实且高度可行的目标。

## 档位 B：可投稿论文

需要额外满足：

- 至少一个稳定的新方法贡献；
- 多 seed/order；
- 多个强 baseline；
- event-wise 和 worst-case 改善；
- 充分消融；
- 不仅在一个事件有效；
- 不是 reject-all；
- 至少一个额外 backbone 或 shift robustness；
- 统计置信区间；
- 清晰的研究完整性记录。

可考虑：

- IGARSS；
- WHISPERS；
- JSTARS；
- GRSL；
- 遥感或灾害相关 workshop/conference。

## 档位 C：TGRS 级完整投稿

除了档位 B，还通常需要：

- 更强的跨数据集或跨任务验证；
- 更多 backbone；
- 更全面的 SOTA 对比；
- 更完整的物理和方法分析；
- 更成熟的方法，而不仅是简单工程规则；
- 稳定且显著的性能改善；
- 充分解释何时成功、何时失败；
- 代码和复现材料成熟。

可以把 TGRS 作为目标，但不能现在承诺一定达到。

---

# 十五、如果实验结果不理想，如何安全转向

## 情况 1：自训练模型上 TTA 仍然明显崩溃

继续 ChangeGuard，这是最理想的研究路径。

## 情况 2：自训练模型上 TTA 不再崩溃

说明之前的崩溃可能与官方 checkpoint 或预处理不确定性有关。

转向：

- checkpoint provenance；
- preprocessing sensitivity；
- reproducibility failure；
- controlled versus black-box baseline comparison。

## 情况 3：所有 TTA 都比 source-only 差

停止 controller。

改为：

- SAR flood TTA failure characterization；
- negative-transfer analysis；
- event-wise benchmark；
- source-anchored conservative adaptation。

## 情况 4：consistency 方法有效，但 controller 无效

以 consistency/anchoring 为主贡献，controller 作为负面实验。

## 情况 5：方法只在 Event 559 有效

不能声称通用方法。只能：

- 降低论文主张；
- 研究 event suitability；
- 说明什么类型事件可适应；
- 不把单事件结果包装为整体成功。

---

# 十六、从现在开始的第一条行动

当前不要直接让 Antigravity 启动四个正式训练。

首先完成：

1. 固定 22/5 source-inner split；
2. 写清 `[VV,VH]` 与 `[VV,VH,slope]` 两套 source configs；
3. 审查 slope 的对齐和 normalization；
4. 实现 resume、manifest 和 completeness；
5. 跑每种输入的短 smoke test；
6. 确认无 NaN、mask 正确、loss 会下降；
7. 用户本人查看 `nvidia-smi`；
8. 只在确认空闲 GPU 后启动第一轮 seed 0；
9. 同时开始写 11 月 10 日的 Interim Report。

这一步完成后，才进入长期 source training。