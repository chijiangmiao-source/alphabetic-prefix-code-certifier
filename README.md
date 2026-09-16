# 深度受限字母序前缀码服务（Depth-Limited Alphabetic Prefix Code）

纯后端服务。给定 **按数组顺序固定的唯一 ASCII 符号**、各自的整数权重（1–10⁹）
与最大码长 `D`（0–15，符号数 n 为 1–40），求解：

> 所有叶子**自左向右严格保持输入顺序**、每个内部节点恰有两个孩子、
> 叶深不超过 `D` 的前缀树，使 **总代价 C = Σ weight[i] · depth[i] 最小**。

- 左孩子编码 `0`，右孩子编码 `1`；
- **单符号树**深度为 0、码字为空串 `""`、代价 0、见证为空；
- 若 `n > 2^D` 则不存在可行树，返回 `infeasible`，**不携带任何部分码表**；
- 最优树数量只报告到 **2**：`unique`（唯一）或 `multiple`（至少两棵）；
  多棵最优树时，按**内部节点前序记录的分界下标整数序列**取**字典序最小**者
  作为规范见证（canonical witness）。

技术栈：Python 3.13 · FastAPI · Pydantic v2 · pytest · Docker Compose。

---

## 1. 运行

### Docker Compose（推荐）

```bash
docker compose up --build            # 默认宿主端口 8000
API_PORT=9099 docker compose up --build
```

宿主端口由环境变量 **`API_PORT`** 覆盖（容器内固定 8000）。

### 一次性验收服务 `verify`

镜像内置完整测试套件（含 Catalan 全树穷举复核）。运行结束即退出，
退出码即验收结论：

```bash
docker compose run --rm verify
```

### 本地直接运行

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload
.venv/bin/pytest -v
```

服务启动后：

- 健康检查：`GET /health` → `{"status":"ok"}`
- 交互文档：`http://localhost:8000/docs`

---

## 2. 接口

### `POST /api/v1/encode`

请求体：

| 字段        | 类型         | 约束                                   |
| ----------- | ------------ | -------------------------------------- |
| `symbols`   | `string[]`   | 长度 1–40；每个是**单个 ASCII 字符**；数组内**唯一** |
| `weights`   | `integer[]`  | 长度 1–40；与 `symbols` **等长并按下标对齐**；每项 1–10⁹ |
| `max_depth` | `integer`    | 0–15                                   |

未知字段一律拒绝。

#### 成功（可行）— `200`

```json
{
  "status": "optimal",
  "cost": 14,
  "optimality": "unique",
  "codes": ["00", "01", "10", "11"],
  "depths": [2, 2, 2, 2],
  "splits": [1, 0, 2],
  "assignments": [
    {"index": 0, "symbol": "A", "code": "00", "depth": 2}
  ]
}
```

- `cost`：最小**整数**代价；
- `codes[i]` / `depths[i]`：第 `i` 个输入符号的码字与叶深；
- `splits`：**树的前序区间分割序列**。对区间 `[i..j]`，取序列中下一个分界 `k`，
  左子树覆盖 `[i..k]`（补码位 `0`），右子树覆盖 `[k+1..j]`（补码位 `1`），
  递归即得整棵树。长度恒为 `n-1`（单符号时为 `[]`）。**每个码字都能仅凭该序列重放**；
- `optimality`：`unique` 或 `multiple`（封顶到 2）。

多解示例：`symbols=["X","Y","Z"], weights=[1,1,1], D=2`。根分界取 0 或 1
代价同为 5，服务返回 `multiple`，并选字典序更小的见证 `[0,1]`
（码字 `0 / 10 / 11`）。

#### 不可行 — `200`

当 `n > 2^D`：

```json
{
  "status": "infeasible",
  "reason": "no alphabetic prefix tree exists with 5 leaves at depth <= 2; ...",
  "capacity": 4
}
```

响应**只含** `status / reason / capacity`，绝不泄露码表、深度、分割或代价。

#### 请求错误 — `422`

**所有**校验错误会被聚合、**按 JSON Pointer（RFC 6901）排序后一次返回**：

```json
{
  "errors": [
    {"pointer": "/max_depth", "message": "Input should be less than or equal to 15"},
    {"pointer": "/symbols/1", "message": "duplicate symbol 'A'; first occurrence at /symbols/0"},
    {"pointer": "/symbols/3", "message": "symbol must be ASCII (code <= 127), got U+00E9"},
    {"pointer": "/weights", "message": "weights length 2 must equal symbols length 4"}
  ]
}
```

根级问题（空体、非法 JSON、非对象体）使用空指针 `""`。

### 快速尝试

```bash
curl -s localhost:8000/api/v1/encode -H 'content-type: application/json' \
  -d '{"symbols":["A","B","C","D"],"weights":[2,2,1,2],"max_depth":15}'
```

---

## 3. 算法

逐次合并最低权重（Huffman / 相邻最小合并）**不能**用于本题：它既会破坏叶序，
也可能产生超过 `D` 的链状树，还会错过代价更低的保序树（见下）。

采用 **区间动态规划（精确整数求解，无浮点）**：

- 状态 `dp[h][(i,j)]`：叶子区间 `[i..j]` 在「至多还可用 h 层边」时的最优结果
  `(cost, count, signature)`；
- 单叶：任意 `h≥0` 代价 0；
- 内部节点枚举分界 `k∈[i,j-1]`，左右子树各用 `h-1` 层：

  ```
  cost = dp[h-1][i,k].cost + dp[h-1][k+1,j].cost + Σ_{t=i..j} weight[t]
  sig  = (k,) + left.sig + right.sig
  ```

  （新内部节点给区间内每个叶子贡献一层。）
- 计数封顶为 2：`min(2, 左数 × 右数)`，不同分界同代价时相加再封顶；
- 等代价时保留 `signature` 字典序最小者；
- 容量剪枝：区间叶子数 `> 2^h` 直接不可行；`n > 2^D` 立即判 `infeasible`。

复杂度 O(D·n³) 时间、O(D·n²) 空间；n=40、D=15 上限规模在亚秒级完成。

响应组装时**不信任 DP 内部展开的码字**：只用 `splits` 经 `replay()` 重放出
码字与深度，并独立复核深度上限、前缀无关性、叶序一致性与代价总和。

### 贪心陷阱见证

- `weights=[2,2,1,2], D=15`：左优先的相邻最小合并得代价 **15**，
  保序最优平衡树代价 **14**（见证 `[1,0,2]`）；
- `weights=[100,1,1,1], D=2`：贪心把三个 `1` 串成高 3 的链（违反 D=2），
  受限最优代价 **206**；
- 容量临界：`n = 2^D` 时全满平衡树是唯一可行形状，`n = 2^D + 1` 立即不可行。

---

## 4. 测试

```bash
.venv/bin/pytest -v          # 或 docker compose run --rm verify
```

- `tests/test_bruteforce_crosscheck.py`
  - 随机 + 地毯式 **Catalan 全树穷举**，逐一复核最优代价、**精确多解数量**、
    字典序最小见证（n≤7）；
  - 每个见证都能 `replay` 重放，并校验前缀无关、叶序、Kraft 和、代价；
  - 损坏见证（越界 / 残余 / 缺失分界）必须被拒绝。
- `tests/test_witnesses.py`：贪心陷阱、深度强制重平衡、容量临界、单符号、
  多解 canonical、唯一坍缩、10⁹ 权重精确整数、n=40/D=15 性能。
- `tests/test_api.py`：端到端成功/不可行（无部分树泄露）、错误按
  JSON Pointer 聚合排序且一次返回、非法 JSON / 空体 / 非对象 / 未知字段 / 边界。

## 5. 目录

```
app/
  solver.py       # 区间 DP + 见证重放（无占位实现）
  schemas.py      # Pydantic 请求/响应/错误模型
  validation.py   # 错误聚合、JSON Pointer 定位、排序
  service.py      # 求解 → 重放 → 独立复核 → 响应组装
  main.py         # FastAPI 路由与异常处理
tests/            # 穷举复核 + 见证 + API
Dockerfile        # python:3.13，默认跑 API
docker-compose.yml# api（API_PORT 覆盖宿主端口）+ 一次性 verify 服务
```
