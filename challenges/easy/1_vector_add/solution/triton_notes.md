# Triton 编程模型笔记（以向量加法为例）

参考代码：`solution.py`（向量加法 vector_add）

## 核心思想：以「Block（块）」为单位编程

Triton 和 CUDA 最大的区别在于**并行的粒度不同**：

- **CUDA**：写的是「单个 thread（线程）」的视角。每个线程处理 1 个元素，需要手动管理 `threadIdx`、`blockIdx`、线程同步、shared memory 等细节。
- **Triton**：写的是「单个 program（也叫 block / tile）」的视角。每个 program 一次性处理**一整块数据**（如 1024 个元素），块内的线程调度、向量化、访存合并等都由 Triton 编译器自动完成。

所以 Triton 中操作的天然是「数组 / 向量」，而不是标量。这就是为什么 `tl.load` / `tl.store` 一次读写一整段。

## 逐行拆解

### 1. Host 端：启动 kernel（`solve` 函数）

```python
BLOCK_SIZE = 1024
grid = (triton.cdiv(N, BLOCK_SIZE),)
vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)
```

- `BLOCK_SIZE = 1024`：每个 program 负责 1024 个元素。
- `grid = (triton.cdiv(N, BLOCK_SIZE),)`：一共启动多少个 program。`cdiv` 是**向上取整除法**，保证即使 `N` 不能被 1024 整除也能覆盖所有元素。例如 `N = 25,000,000` → 需要 `24415` 个 program。
- `vector_add_kernel[grid](...)`：`[grid]` 语法告诉 Triton 按此网格规模并行启动 kernel，类似 CUDA 的 `<<<grid, block>>>`，但只需指定 program 的数量。

### 2. Device 端：kernel（`@triton.jit`）

```python
@triton.jit
def vector_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a_data = tl.load(a + offsets, mask=mask)
    b_data = tl.load(b + offsets, mask=mask)
    c_data = a_data + b_data
    tl.store(c + offsets, c_data, mask=mask)
```

- **`@triton.jit`**：即时编译（Just-In-Time）。函数首次调用时被编译成 GPU 机器码。
- **`BLOCK_SIZE: tl.constexpr`**：`constexpr` 表示**编译期常量**。编译器知道具体值后可做循环展开、向量化等优化。这也是 `BLOCK_SIZE` 要作为参数传入而非硬编码的原因。
- **`pid = tl.program_id(axis=0)`**：取得「我是第几个 program」的编号，用于区分各 program 负责哪块数据。
- **`block_start = pid * BLOCK_SIZE`**：本 program 负责的数据起始位置。例如 `pid = 2` → 从第 2048 个元素开始。
- **`offsets = block_start + tl.arange(0, BLOCK_SIZE)`**：生成长度为 `BLOCK_SIZE` 的**索引向量**（如 `[2048, 2049, ..., 3071]`）。这是 Triton「块式」思维的核心——一次生成一整组下标。

### 3. 为什么需要 `mask`（掩码 / 边界保护）

因为 `N` 通常不是 `BLOCK_SIZE` 的整数倍，**最后一个 program 会越界**。

- `mask = offsets < n_elements`：标记哪些下标合法（`True`）、哪些越界（`False`）。
- `tl.load(..., mask=mask)`：只加载合法位置，越界位置不读内存（避免非法访存 / 段错误）。
- `tl.store(..., mask=mask)`：只写回合法位置，防止污染 `c` 之外的内存。

这就是 **boundary check（边界保护）**，几乎是每个 Triton kernel 的标准写法。

## 与 CUDA 写法对照

同样的向量加法，CUDA 是单线程视角：

```cuda
__global__ void add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 单个元素
    if (i < n) c[i] = a[i] + b[i];                  // 标量操作 + 标量边界判断
}
```

| 维度 | CUDA | Triton |
|------|------|--------|
| 编程视角 | 单个 thread（标量） | 单个 program / block（向量） |
| 索引 | `blockIdx*blockDim + threadIdx`（标量 `i`） | `pid*BLOCK_SIZE + arange`（向量 `offsets`） |
| 边界处理 | `if (i < n)` | `mask = offsets < n_elements` |
| 块内线程管理 | 手动 | 编译器自动 |
| 访存合并 / 向量化 | 手动优化 | 编译器自动 |

## 通用骨架（记住这四步）

Triton 让你「以块为单位、用类 NumPy 的向量语法写 GPU kernel」，把线程级细节交给编译器。绝大多数 Triton kernel 都是这个套路：

1. 用 `program_id` 确定「我是哪一块」；
2. 用 `arange` 生成这一块的下标向量 `offsets`；
3. 用 `mask` 做边界保护；
4. `load → 计算 → store`，全程对整块数据操作。

更复杂的 kernel（reduction、matmul 等）都是在这个 `pid → offsets → mask → load/compute/store` 范式上扩展的。
