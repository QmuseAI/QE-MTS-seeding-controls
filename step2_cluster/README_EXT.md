# 扩展实验（第 3 轮）：代码、运行顺序与时间预估

把本包的 `code/`、`slurm/`、`gpu/` 三个文件夹直接合并进集群上的 `qe_mts/step2_cluster/`。全部是新文件名，不会覆盖任何已有文件；`run_main/` 只读，新结果写到 `run_ext/`。

我只做了编译检查（gcc `-Wall -Wextra` 无警告）、Python 语法检查和 shell 语法检查，**没有运行任何程序**。第一次运行请先看下面每一步的检查项。

## 新文件

| 文件 | 作用 |
|---|---|
| `code/mts4.c` | mts v3 加上可选的 `key=value` 设置：种群大小 K、种子拷贝数 `kcopies`（其余为随机串）、拷贝扰动 `nflip`、`pmut`、`tabu`、无种群的迭代 tabu（`solver=1`）、输出到达的最优解（`cfg=1`）。**不加任何设置时与 v3 逐位相同**（由 V0/V2m 两组验证）。|
| `code/make_tasks_ext.py` | 新组的任务表（定义了全部新组），`--list-missing` 列出缺少的种子文件 |
| `code/analyze_ext.py` | 第 0–9 节分析（见下），输出 `run_ext/analysis_ext/` |
| `slurm/plan_ext.txt` | 第 1 批：N≤42 的全部新组 |
| `slurm/plan_hard.txt` | 第 2 批：A2m 补到 N=43–48，最优解收集到 48 |
| `slurm/plan_dcqo.txt` | 第 3 批：DCQO 种子的三组（GPU 种子生成后再跑）|
| `slurm/19_prep_ext.sbatch`、`slurm/submit_ext.sh`、`slurm/32_analyze_ext.sbatch` | 编译 mts4、补种子、建任务表 → 跑 → 合并与分析 |
| `gpu/dcqo_seeds.py`、`gpu/run_dcqo.sh` | CUDA-Q 生成 DCQO 种子（自检、Trotter 收敛、时间估计、生成）|

## 新组

| 组 | 内容 | N | 回答的问题 |
|---|---|---|---|
| V0, V2m | mts4 不加设置，重跑 A0 / A2m 的 replicate 0–9 | 27–30 | mts4 是否与原程序完全一致 |
| C0 | 与 A0 完全相同的运行（replicate 0–19），并记录到达的最优解 | 27–48 | 最优解简并度、是否斜对称，与奇偶难度的关系 |
| M2m_k1/k10/k50 | A2m 种子 k 份 + (100−k) 个随机串 | 27–42 | 拷贝数如何决定开销 |
| P2m_d8/d4 | A2m 种子 100 份，每份（除第一份）随机翻转 N/8 或 N/4 位 | 27–42 | 只要恢复多样性，开销是否消失 |
| M3_k1, P3_d8 | 同上，用最深的种子 A3 | 27–42 | 修复对最差种子是否也有效 |
| H0_x / H2m_x | K=50、K=200、p_mut=2/N、tabu 长度×2，各一对（随机种群 / A2m 种子）| 27–40 | 换 MTS 参数后开销是否仍在 |
| I0, I2, I2m, I3 | 无种群的迭代 tabu：随机起点 / A2 / A2m / A3 种子起点 | 27–37 | 机制是否来自种群（预期：无种群就没有开销）|
| A2m（补） | 原 A2m 组，用原程序 mts | 43–48 | 和原文定量吻合的那一组，在原文声称的交叉点 N≈47 处是否交叉 |
| Q1, Q5, PQ1_d8 | DCQO 1e5 shots 中最好的 ×100；100 个最好的不同串；DCQO 种子加扰动 | 27–33（或到 36）| 真正的量子种子在同一协议下的表现 |

分析（`analyze_ext.py`）另外三项不需要新数据：把种子成本计入 TTS（第 2 节）、组内逐个 replicate 的种子能量 / Hamming 距离与 TTS 的相关（第 3 节）、奇偶难度（第 4 节）。

## 运行顺序（在集群登录节点，`step2_cluster/` 下）

```bash
bash slurm/submit_ext.sh                                                  # 第 1 批
TAG=hard THREADS=32 PLANX=slurm/plan_hard.txt bash slurm/submit_ext.sh    # 第 2 批（可同时提交，排队即可）
```
GPU 机器上（`gpu/` 文件夹，需要 cudaq、numpy、scipy，以及 `../code/labs_common.py`）：
```bash
bash run_dcqo.sh            # 自检 + N=27 收敛测试 + 时间估计，先看结果
STEPS=16 bash run_dcqo.sh seeds    # STEPS 按收敛测试结果设；N=27–33 每张卡一个 N 并行
```
把 `seeds_dcqo/` 里的 `seeds_Q*_N*.txt` 和 `seed_stats_dcqo.csv` 拷到集群 `step2_cluster/seeds/`，按实际做到的 N 修改 `plan_dcqo.txt` 的上限，然后：
```bash
TAG=dcqo PLANX=slurm/plan_dcqo.txt bash slurm/submit_ext.sh               # 第 3 批
```
每一批结束都会自动重跑一次分析；最后一次的 `run_ext/analysis_ext/` 就是完整结果。把整个 `run_ext/analysis_ext/` 和 `run_ext/data/` 打包传回来即可。

## 时间预估

依据：上次实测每个逻辑核约 50–60 ns 完成一次评估，2000 个工作线程约 1.3×10¹⁴ 次评估 / 小时；各组成本按 run_main 里 A0/A2/A2m/A3 每个 N 的实际评估次数推算。

| 批次 | 评估次数 | 理想时间 | 预计墙钟 |
|---|---|---|---|
| 第 1 批 plan_ext | ≈3×10¹⁴（N=39、41 占大头）| ≈2.5 h | 3–4 h |
| 第 2 批 plan_hard | ≈5×10¹⁴（N=43 约占一半）| ≈4 h | 5–7 h；去掉 N=43 约 1.5–2 h |
| 第 3 批 plan_dcqo | ≈10¹² | 几分钟 | <0.5 h（含排队、分析）|
| 每次分析 | — | — | 约 0.5 h |
| GPU 自检 + 收敛 + 估计 | — | — | 约 0.5 h |
| GPU 种子 N=27–33（7 张卡并行）| — | — | 约 1–3 h（由 N=33 决定，`--estimate` 会给出实际数）|
| GPU 种子 N=34–36（8 卡 mpirun，可选）| — | — | 每个 N 数小时，建议先不做 |

集群合计约 10–12 小时；GPU 与第 1 批可以同时进行。

## 第一次运行时要看的检查项

1. `run_ext/analysis_ext/summary_ext.txt` 第 1 节：V0 与 A0、V2m 与 A2m 必须 **全部 identical**。如果出现 MISMATCH，其他新组的结果先不要用，把日志发我。
2. 第 4 节：每个到达的最优解都会用能量重新核对；出现 `ENERGY CHECK FAILED` 说明 cfg 输出有问题。
3. GPU `logs/selftest.txt`：最后一行应为 `SELFTEST OK`，并给出 `use --sign -1`（或 +1）。若是 +1，生成种子时用 `SIGN=1 STEPS=16 bash run_dcqo.sh seeds`。若出现 `WARNING: no convention matches`，先停下，把日志发我。
4. GPU `logs/converge_N27.txt`：选两个数字都稳定（变化约 0.5% 以内）的最小步数作为 STEPS。

## 关于 DCQO 的实现说明

原文的 CD 项 O1 = [H_i, H_f] 与 λ 无关，冲量区间下 H(t) = λ̇ A(λ) 在所有时刻都正比于同一个算符，所以演化严格等于 exp(Θ O1)，Θ = ∫₀¹ α1(λ) dλ。原文说的“n_trot → ∞”就是这个精确指数；与 λ(t) 的形式和总时间 T 都无关。脚本用 Pauli 串代数精确计算 α1(λ)（不需要 2^N 矩阵），并在小 N 上与稠密矩阵逐项核对。论文方法部分可以直接这样写，原文“n_trot→∞ 与电路深度”之间的矛盾也就不影响我们的复现。
