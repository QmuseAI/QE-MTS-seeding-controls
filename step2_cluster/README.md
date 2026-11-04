# 第 2–3 步：经典种子对照的全规模实验（SLURM，4 节点 × 512 CPU）

日期：2026-10-20。本目录下的代码只做过最基本的检查：gcc `-Wall -Wextra` 编译无警告、Python 语法检查、shell 语法检查，以及最优能量表的核对（见下）。**没有跑过任何 MTS 或分析**，第一次运行请先看 timing 作业的输出。

## v3 改动（2026-10-20，为了把整个实验压到约 1 小时）

- **按 replicate 提前停止（右删失）。** 一个 replicate 的 S 个 run 按逐步加大的评估预算一起推进；一旦最小的 kstop = ⌈0.7·S⌉ 个 TTS 都已确定，其余 run 停止，输出 evals = −(已用评估数)，分析时按 +∞ 处理。每个 run 的轨迹只由 (base, r, s) 决定，暂停后继续不会改变轨迹，所以已确定的 TTS 与 v2 顺序代码逐位相同。replicate 中位数是精确的；对 seed 做 bootstrap 时，重抽样中位数超过第 70 个次序统计量的概率约为 1e−5，两阶段 bootstrap 也不受影响（analyze.py 会检查并报警）。论文里写一句"runs were right-censored once the 70 fastest of 100 seeds had finished"即可。
- **每个任务用 OpenMP 多线程**（默认 THREADS=8，每节点同时跑 500/8 个任务），最慢的 N=50 任务不再是瓶颈。
- **每个 run 的安全上限默认 3e9 次评估**（make_tasks.py `--cap`），只用来防止极端 replicate 拖住整体。
- 估算：`python code/estimate_cost.py --timing run_main/timing --plan slurm/plan.txt --nodes 4 --cpus 500`。
- 续跑：`CAP=... KFRAC=... THREADS=... bash slurm/resubmit_run.sh`（都有默认值）。`KFRAC=1` 恢复 v2 的行为。

## 实验设计：每一组回答审稿人的哪个问题

| 组 | 初始种群 | N 范围 | 回答的问题 |
|---|---|---|---|
| A0 | K=100 个随机串 | 20–50 | 原文的 MTS 基线；检查我们的实现能否复现原文 κ≈1.37 和 TTS 的绝对量级（与第 1 步提取的 Fig.2 逐 N 对比） |
| A1 | 1 个随机串 ×100 | 20–50 | 零信息种子：单靠“复制一个串”这个协议，能不能让斜率变平？ |
| A2 | 1e5 个均匀样本里能量最低的 1 个 ×100 | 20–50 | 原文 QE-MTS 协议，只把量子采样换成均匀采样，shot 数相同 |
| A5 | 1e5 个均匀样本里能量最低的 100 个**不同**串 | 20–45 | 预算相同但保持多样性：区分“种子质量”和“多样性坍缩” |
| A6 | 1e7 个均匀样本里能量最低的 100 个不同串 | 27–40 | 更大的采样预算（对应原文 App. D 的“多次 DCQO 运行”） |
| A7a / A7b | 已知最优解随机翻转 N/4 / N/8 位 ×100 | 27–40 | 神谕种子，质量远好于任何采样器：回应“DCQO 种子可能比 A2 好得多” |
| A0na / A2na | 同 A0 / A2，但 tabu 不用 aspiration | 27–40 | 原文 Alg.2 伪代码没有 aspiration，App. F 文字里却有。A0na/A2na 与 A0/A2 用相同的随机流，可以配对比较 |

每组都是 100 replicates × 100 seeds，与原文 Sec. 3.3 相同。统计流程完全照原文（每个 replicate 先对 seed 取中位数，再对 replicate 取分位数；log Q 对 N 线性拟合；两阶段 bootstrap，B=2000；交叉点按原文 Eq.10），并额外报告：

- 0. 实现检查：我们的 A0 与原文 MTS 逐 N 的 log10 差、相关系数、κ。
- 1. 各组 κ：paper 窗口、全范围、33–37、>37。同时给 bootstrap CI 和含逐 N 失拟的 OLS CI。
- 2. 配对差距 D_x(N) 及 bootstrap CI；κ 比值；与原文 Fig.5 的 QE-MTS 差距直接对比（RMS、相关）。
- 3. 用 27–37 窗口外推交叉点（原文 Eq.10，以及 Q50 直线的交点），再用 N>37 的实测数据**直接检验**是否真的交叉。
- 4. D 的尾部：用 33–37 拟合线性与几何衰减两种模型，在 N>37 上做样本外检验。
- 5. aspiration 开/关的敏感性。
- 6. 种子质量：种子能量与最优能量之比、到已知最优解的 Hamming 距离。

## 代码修改（相对第 1 版 mts.c）

- 随机数改为 xoshiro256**，由 splitmix64 从 (base, replicate, seed) 初始化，各条流相互独立。
- 读种子用 `SCNu64`，种子数组动态分配，fopen 失败会报错。
- 新增 mode 2：每个 replicate 读入 K 个不同的串（A5/A6）。
- 新增命令行参数：aspiration 开关、是否计入初始种群的评估次数；默认 cap 改为 1e12，实际上不设上限。
- 计数规则不变：一次 energy() 或一次 flipE() 记 1 次评估；与原文一样，生成种子的成本不计入。

## 最优能量

N=20–50 的最优能量和最优序列取自 Packebusch & Mertens, J. Phys. A 49, 165001 (2016)，即 arXiv:1512.02475 的表 1–2（穷举，N≤66 精确）。`verify_eopt.py` 会从游程编码的序列重新计算能量，并检查 8 个对称像，**31 个 N 已全部核对通过**。prep 作业里还会用 MTS 以“最优能量 − 4”为目标跑一遍，作为独立检查：它必须永远找不到。

## 运行步骤

SLURM 脚本的写法与 `QEC_SB_paper1/code/slurm` 保持一致：
- 环境：`module load Core/anaconda3-2024.02-1`，conda 环境沿用 `qecsb`（只需要 numpy、scipy、matplotlib；换环境可设 `CONDA_ENV=...`）。
- 节点：用 sinfo 自动统计分区里所有可用节点（包括 n013，不排除任何节点），数组大小 N 等于节点数。
- 每个节点：`--exclusive --cpus-per-task=512 --mem=0`，但只开 **500 个工作进程**，留 12 个逻辑 CPU 空着（可用 `WORKERS=...` 修改）。
- 日志写在 `logs/`，所有脚本都从项目根目录（step2_cluster）提交。

把整个 `qe_mts` 目录（含 `step1_digitize/` 和 `step2_cluster/`）拷到集群上，然后在登录节点执行：

```bash
cd qe_mts/step2_cluster
bash slurm/submit_all.sh      # 一次提交整条链：prep -> timing -> run(数组，每节点一个) -> analyze
squeue -u $USER
cat run_main/timing/estimate.txt     # timing 跑完后可查看预计的核时和墙钟时间
```

可选参数（与 QEC 那边相同）：`N=4`（固定节点数）、`EXCLUDE=n013`（需要时排除某个节点）、`EXTRA="--partition=xxx"`、`WORKERS=500`、`SKIP_TIMING=1`。

- **中断后续跑**：`bash slurm/resubmit_run.sh`。它会重建只含缺失任务的任务表，再提交 run 数组和分析。种子是确定性的，不需要重新生成。
- **结果位置**：`run_main/analysis/` 下有 summary.txt、perN_quantiles.csv、kappa_table.csv、crossover_tests.csv、fig_scaling_and_gap.png。原始数据在 `run_main/data/`，环境与哈希值在 `run_main/provenance.txt`。

## 计算量（粗估，以 timing 作业的输出为准）

- 我的粗估：主体（A0/A1/A2 跑到 N=50）约 700 核时，其余各组合计约 100–200 核时。
- 以 4 个节点 × 500 进程计，理想墙钟时间不到 1 小时（以 submit_all.sh 打印的 N 为准）。实际受限于 N=50 的单个任务（100 个 seed 串行），预计 1–3 小时。
- 20_run 的 `--time=24:00:00` 留了很大余量。
- 如果 timing 的估计远超预期，可以把 plan.txt 里 A0/A1/A2 的上限从 50 改成 47。

## 已知局限（写文章时如实说明）

- 没有作者的 DCQO 种子。A4（GPU 上的 DCQO 态矢量模拟）只能做到 N≈35 左右，而且 n_trot 的设置原文自相矛盾。本实验用 A6/A7 覆盖“种子更好”的情形。
- A7 只用了每个 N 的一个已知最优解（及其 8 个对称像）。
- 只有一个 LABS 实例/每个 N。这是问题本身的性质，原文也一样；我们因此报告配对差距和含失拟的区间。
