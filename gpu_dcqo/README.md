# gpu_dcqo：DCQO 种子独立包（GPU 服务器用）

把整个 `gpu_dcqo/` 文件夹上传到 GPU 服务器，在文件夹里运行即可，不依赖集群上的任何文件（`labs_common.py` 已经放在包里）。
结果都写在包内：`seeds_dcqo/`（种子与统计）、`logs/`（日志）。

## 环境
- Python 3.10+，`numpy`、`scipy`，以及 CUDA-Q：`pip install cudaq`（RTX PRO 6000 是 Blackwell 架构，请装最新版 CUDA-Q 和对应驱动）。
- N=34–36 需要多卡（`big` 步骤，mpirun + CUDA-Q 的 MPI 支持）；N≤33 单卡即可，不需要 MPI。

## 步骤
```bash
bash run_dcqo.sh env                 # 0. 检查 python / cudaq / GPU，约 1 分钟
bash run_dcqo.sh                     # 1-3. 自检 + N=27 收敛测试 + 时间估计，约 30 分钟
```
看三个日志后再继续（也可以把它们发给我）：
- `logs/selftest.txt`：最后应为 `SELFTEST OK`，并提示 `use --sign -1` 或 `+1`。若为 +1，用 `SIGN=1 bash run_dcqo.sh` 重跑第 1–3 步（收敛测试依赖这个符号），之后的命令也都加 `SIGN=1`。若出现 `WARNING: no convention matches`，先停下。
- `logs/converge_N27.txt`：两个数字随步数变化，选它们稳定（变化约 0.5% 以内）的最小步数作为 STEPS。
- `logs/estimate.txt`：每个 N 每个 Trotter 步的预计分钟数；乘以 STEPS 就是该 N 的时间。

```bash
STEPS=16 bash run_dcqo.sh seeds      # 4. N=27–33，每张卡一个 N 并行，约 1–3 小时（由 N=33 决定）
bash run_dcqo.sh pack                # 5. 打包成 dcqo_results.tar.gz
```
可选：`STEPS=16 bash run_dcqo.sh big` 生成 N=34–36（8 卡并行，每个 N 数小时）。

## 结果怎么用
把 `dcqo_results.tar.gz` 发给我，或者把 `seeds_dcqo/` 里的
`seeds_Q1_N*.txt`、`seeds_Q5_N*.txt`、`seed_stats_dcqo.csv` 拷到集群 `step2_cluster/seeds/`，
按实际做到的 N 改 `slurm/plan_dcqo.txt` 的上限，然后在集群上：
```bash
TAG=dcqo PLANX=slurm/plan_dcqo.txt bash slurm/submit_ext.sh
```

## 文件
| 文件 | 作用 |
|---|---|
| `dcqo_seeds.py` | DCQO 模拟与采样：精确计算 Θ（Pauli 串代数）、小 N 稠密矩阵自检、Trotter 收敛、时间估计、生成种子 |
| `labs_common.py` | LABS 能量、最优能量与最优序列（与集群上 `code/labs_common.py` 相同） |
| `run_dcqo.sh` | 上面各步骤的入口 |

物理说明：原文的 CD 算符 O1 = [H_i, H_f] 与 λ 无关，冲量区间下演化严格为 exp(Θ O1)，Θ = ∫₀¹ α1(λ) dλ，与 λ(t) 和总时间无关；原文的 “n_trot → ∞” 就是这个精确指数，这里用 STEPS 步 Trotter 逼近。
