"""深度受限的字母序最优二叉前缀树求解器。

约束（题目固定）：
* n 个叶子自左向右严格保持输入下标顺序 0..n-1；
* 每个内部节点恰有两个孩子，左孩子代表码位 0，右孩子代表码位 1；
* 任意叶子深度不超过 D；
* 目标：最小化 C = Σ weight[i] * depth[i]（整数代价，精确求解）。

状态定义
--------
dp[h][(i, j)] 表示叶子区间 [i..j] 在「至多还可使用 h 层边」时的最优结果：
(cost, count_capped, signature)。

* 长度为 1 的区间在任意 h>=0 下都是代价 0 的叶子；
* 长度 >=2 的区间在 h>=1 时枚举分界 k∈[i, j-1]：左 [i..k]、右 [k+1..j]，
  两个子树都只拿到 h-1 层预算；
  该内部节点给区间内每个叶子贡献一层，故代价加 Σ_{t=i..j} weight[t]；
* 分界、左子树、右子树都可行时候选才可行；
* 最优树数量只保留到封顶值 2：count = min(2, 左数 * 右数)，
  同代价的不同分界之间再相加，仍封顶为 2；
* signature 是「内部节点前序遍历记录的分界下标整数序列」，
  即 (k,) + 左子 signature + 右子 signature。等代价时取字典序最小者，
  这正是题目要求的规范（canonical）见证选择。

n<=40、D<=15 时状态数 O(D*n^2)、转移 O(n)，整体 O(D*n^3)，可在亚秒级完成。
"""

from __future__ import annotations

from dataclasses import dataclass

COUNT_CAP = 2
"""最优树计数的封顶值：达到 2 即表示「至少两棵」。"""


@dataclass(frozen=True)
class OptimalTree:
    """一棵可行实例的最优结果。"""

    cost: int
    """最小整数代价 Σ weight[i]*depth[i]。"""

    splits: tuple[int, ...]
    """内部节点前序记录的分界下标序列（长度 n-1，单符号时为空）。"""

    count_capped: int
    """最优树数量，封顶为 2（1 表示唯一，2 表示至少两棵）。"""

    @property
    def optimality(self) -> str:
        return "unique" if self.count_capped == 1 else "multiple"


def solve(weights: list[int], max_depth: int) -> OptimalTree | None:
    """求深度受限的字母序最优前缀树；不可行时返回 None（不泄露任何部分树）。"""
    n = len(weights)
    capacity = 1 << max_depth
    if n > capacity:
        return None

    prefix = [0]
    for w in weights:
        prefix.append(prefix[-1] + w)

    # tables[h][(i, j)] -> (cost, count, signature) 或 None（该层预算下不可行）
    tables: list[dict[tuple[int, int], tuple[int, int, tuple[int, ...]] | None]] = [
        {} for _ in range(max_depth + 1)
    ]

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            interval_weight = prefix[j + 1] - prefix[i]
            key = (i, j)

            # h == 0：只容得下单叶。
            tables[0][key] = (0, 1, ()) if length == 1 else None

            for h in range(1, max_depth + 1):
                if length == 1:
                    tables[h][key] = (0, 1, ())
                    continue
                # h 层满二叉树至多容纳 2^h 个叶子，容量不足直接判不可行。
                if length > (1 << h):
                    tables[h][key] = None
                    continue

                best_cost: int | None = None
                best_count = 0
                best_signature: tuple[int, ...] | None = None
                child_table = tables[h - 1]

                for k in range(i, j):
                    left = child_table.get((i, k))
                    right = child_table.get((k + 1, j))
                    if left is None or right is None:
                        continue
                    cost = left[0] + right[0] + interval_weight
                    count = min(COUNT_CAP, left[1] * right[1])
                    signature = (k,) + left[2] + right[2]

                    if best_cost is None or cost < best_cost:
                        best_cost, best_count, best_signature = cost, count, signature
                    elif cost == best_cost:
                        best_count = min(COUNT_CAP, best_count + count)
                        if signature < best_signature:
                            best_signature = signature

                tables[h][key] = (
                    None
                    if best_signature is None
                    else (best_cost, best_count, best_signature)
                )

    entry = tables[max_depth].get((0, n - 1))
    if entry is None:
        return None
    return OptimalTree(cost=entry[0], splits=entry[2], count_capped=entry[1])


def replay(splits: tuple[int, ...], n: int) -> tuple[list[str], list[int]]:
    """仅凭分界见证序列重放整棵树，返回按叶子下标对齐的码字与深度。

    重放规则：对区间 [i..j]，按前序取下一个分界 k——左子树覆盖 [i..k]
    （追加码位 0），右子树覆盖 [k+1..j]（追加码位 1）；区间退化为单叶时，
    当前前缀即该叶码字。见证与树必须一一对应，任何越界或残余分界都视为非法。
    """
    positions = iter(splits)
    bits = [""] * n
    depths = [0] * n

    def build(i: int, j: int, prefix: str) -> None:
        if i == j:
            bits[i] = prefix
            depths[i] = len(prefix)
            return
        try:
            k = next(positions)
        except StopIteration:
            raise ValueError("splits 序列过短，无法还原内部节点") from None
        if not (i <= k < j):
            raise ValueError(f"非法分界 k={k}：不在区间 [{i},{j}) 内")
        build(i, k, prefix + "0")
        build(k + 1, j, prefix + "1")

    if n >= 1:
        build(0, n - 1, "")
    leftover = next(positions, None)
    if leftover is not None:
        raise ValueError("splits 序列存在未消费的残余分界")
    return bits, depths
