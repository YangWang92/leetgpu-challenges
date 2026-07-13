# Local Judge / 本地判题

**[English](#local-judge-english) · [中文](#本地判题中文)**

<a name="local-judge-english"></a>

## Local Judge (English)

Run these challenges on your own GPU and get automatic correctness + performance
results — **without** connecting to the leetgpu.com backend. Grading uses each
challenge's built-in `reference_impl` and test cases, so no remote service or API
key (as `run_challenge.py` needs) is required.

Supported languages: **cuda / triton / pytorch**.

## Contents

| File | Purpose |
| --- | --- |
| `local_runner.py` | The judge. Loads a `challenge.py`, runs your solution, compares against the reference, and times it. |
| `judge.sh` | Launcher. Activates the venv, puts `nvcc` on `PATH`, then calls `local_runner.py`. |
| `new_solution.sh` | Scaffolds a `solution/` file for a challenge/language from its starter template. |

## One-time setup

A CUDA-enabled PyTorch is required (this machine is Blackwell, `sm_120`). Create a
Python 3.12 venv with `uv`:

```bash
cd ~/work/leetgpu-challenges
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch numpy --torch-backend=auto   # auto-selects the matching CUDA wheel
```

The `cuda` language also needs `nvcc` (installed here at `/usr/local/cuda-13.3/bin`).
`judge.sh` adds it to `PATH` automatically; override with `CUDA_HOME` if needed.

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
# expected: True (12, 0)
```

## Usage

### Option A — launcher script (recommended; handles venv + nvcc PATH)

```bash
# judge: <challenge_dir> <language> [test|perf|all]
scripts/judge.sh challenges/easy/1_vector_add cuda
scripts/judge.sh challenges/easy/1_vector_add triton perf
scripts/judge.sh challenges/easy/1_vector_add pytorch all
```

Scaffold a solution from the starter template:

```bash
scripts/new_solution.sh challenges/easy/1_vector_add triton
# -> creates solution/solution.py; implement solve() then run judge.sh
```

### Option B — call Python directly

```bash
source .venv/bin/activate
export PATH=/usr/local/cuda-13.3/bin:$PATH
python scripts/local_runner.py challenges/easy/1_vector_add --language cuda --action all
```

Useful flags: `--solution <path>` (explicit solution file), `--arch <sm_xx>`
(default `sm_120`), `--device <dev>` (default `cuda`).

## Solution file convention

| Language | Path | Entry point |
| --- | --- | --- |
| cuda | `<challenge>/solution/solution.cu` | `extern "C" void solve(...)` |
| triton | `<challenge>/solution/solution.py` | `def solve(...)` |
| pytorch | `<challenge>/solution/solution.pytorch.py` | `def solve(...)` |

Or pass any path with `--solution`. `solution/` is git-ignored, so your solutions
are never committed by accident.

## Actions

- `test` — run every case from `generate_functional_test()`, comparing against the
  reference with the challenge's `atol` / `rtol`.
- `perf` — run the large `generate_performance_test()` case: verify correctness,
  then time it (mean over 50 iterations).
- `all` — both of the above.

## How it works

- **pytorch / triton**: import the solution's `solve()` and call it with the test
  tensors, passed **positionally** in `get_solve_signature()` order (Triton's `solve`
  often uses lowercase parameter names, so positional — not keyword — args are used).
  The output tensors are then compared with those produced by `reference_impl`.
- **cuda**: compile with `nvcc -shared -Xcompiler -fPIC -arch=sm_120` into a `.so`,
  load it via `ctypes`, and call `solve`. Tensors are passed as device pointers
  (`tensor.data_ptr()`); scalars are passed by their declared `ctype`.

## Notes

- Each challenge's `generate_performance_test()` is sized for the online Tesla T4
  (16 GB); this machine's 24 GB is more forgiving. Correctness thresholds use the
  challenge's own `atol` / `rtol`, matching the online judge.
- This local judge covers correctness and timing for cuda/triton/pytorch. It does
  **not** replicate the online platform's leaderboard or multi-GPU features — the
  grading backend for those is not part of this repository (it is closed source, and
  the content is licensed CC BY-NC-ND).

---

<a name="本地判题中文"></a>

## 本地判题（中文）

在**自己的 GPU** 上跑这些题，自动判对错、测性能——**不需要**连接 leetgpu.com 后端。
判题依据是每道题自带的 `reference_impl` 和测试用例，因此不需要 `run_challenge.py`
那套远端服务和 API key。

支持语言：**cuda / triton / pytorch**。

### 文件说明

| 文件 | 作用 |
| --- | --- |
| `local_runner.py` | 判题器。加载 `challenge.py`，跑你的解法，与参考实现对拍并计时。 |
| `judge.sh` | 启动脚本。自动激活 venv、把 `nvcc` 加进 `PATH`，再调用 `local_runner.py`。 |
| `new_solution.sh` | 从 starter 模板为某题某语言初始化 `solution/` 文件。 |

### 一次性环境搭建

需要带 CUDA 的 PyTorch（本机是 Blackwell，`sm_120`）。用 `uv` 建一个 Python 3.12 venv：

```bash
cd ~/work/leetgpu-challenges
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch numpy --torch-backend=auto   # 自动选匹配本机的 CUDA wheel
```

`cuda` 语言还需要 `nvcc`（本机在 `/usr/local/cuda-13.3/bin`）。`judge.sh` 会自动把它
加进 `PATH`；需要时可用环境变量 `CUDA_HOME` 覆盖。

验证：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
# 预期输出: True (12, 0)
```

### 用法

#### 方式一 —— 启动脚本（推荐，自动处理 venv 和 nvcc PATH）

```bash
# 判题: <题目目录> <语言> [test|perf|all]
scripts/judge.sh challenges/easy/1_vector_add cuda
scripts/judge.sh challenges/easy/1_vector_add triton perf
scripts/judge.sh challenges/easy/1_vector_add pytorch all
```

从 starter 模板初始化解法文件：

```bash
scripts/new_solution.sh challenges/easy/1_vector_add triton
# -> 生成 solution/solution.py；实现里面的 solve() 后用 judge.sh 判题
```

#### 方式二 —— 直接调用 Python

```bash
source .venv/bin/activate
export PATH=/usr/local/cuda-13.3/bin:$PATH
python scripts/local_runner.py challenges/easy/1_vector_add --language cuda --action all
```

常用参数：`--solution <路径>`（显式指定解法文件）、`--arch <sm_xx>`（默认 `sm_120`）、
`--device <设备>`（默认 `cuda`）。

### 解法文件约定

| 语言 | 路径 | 入口 |
| --- | --- | --- |
| cuda | `<题目>/solution/solution.cu` | `extern "C" void solve(...)` |
| triton | `<题目>/solution/solution.py` | `def solve(...)` |
| pytorch | `<题目>/solution/solution.pytorch.py` | `def solve(...)` |

也可以用 `--solution` 指定任意路径。`solution/` 已被 git 忽略，你的解法不会被误提交。

### action 说明

- `test` —— 跑 `generate_functional_test()` 的全部用例，按题目的 `atol` / `rtol` 对拍。
- `perf` —— 跑 `generate_performance_test()` 的大用例：先验证正确性，再计时（50 次迭代取平均）。
- `all` —— 以上两者都跑。

### 原理

- **pytorch / triton**：import 解法的 `solve()`，按 `get_solve_signature()` 的顺序
  **位置传参**（Triton 的 `solve` 形参名常是小写，所以用位置而非关键字），再把输出张量
  与 `reference_impl` 的结果对拍。
- **cuda**：用 `nvcc -shared -Xcompiler -fPIC -arch=sm_120` 编成 `.so`，通过 `ctypes`
  加载并调用 `solve`。张量按设备指针（`tensor.data_ptr()`）传入，标量按其声明的 `ctype` 传入。

### 注意

- 每道题的 `generate_performance_test()` 规模是按线上 Tesla T4（16 GB）设计的；本机 24 GB
  更宽松。判定阈值用题目自带的 `atol` / `rtol`，与线上一致。
- 本地判题器覆盖了 cuda/triton/pytorch 的正确性与计时，但**不**复刻线上平台的排行榜、
  多 GPU 等功能——那套评测后端不在本仓库里（闭源，且内容采用 CC BY-NC-ND 许可）。
