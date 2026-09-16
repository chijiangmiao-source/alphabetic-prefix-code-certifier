"""规范见证：贪心陷阱、深度容量临界、单符号、多解 canonical、极值与性能。"""

from __future__ import annotations

import time

from app.solver import replay, solve


def test_greedy_trap_unconstrained() -> None:
    """逐次合并最低权重相邻对会错过总代价更低的保序树。

    [2,2,1,2]：左优先的相邻最小合并得到 15；保序最优是平衡树，代价 14。
    """
    weights = [2, 2, 1, 2]
    result = solve(weights, 15)
    assert result is not None
    assert result.cost == 14
    assert result.optimality == "unique"
    # 平衡见证：根分界 1，左区间 [0,1] 分界 0，右区间 [2,3] 分界 2。
    assert result.splits == (1, 0, 2)
    codes, depths = replay(result.splits, len(weights))
    assert codes == ["00", "01", "10", "11"]
    assert depths == [2, 2, 2, 2]


def test_greedy_trap_depth_forced() -> None:
    """相邻最小合并出的树高违反 D=2，必须重新平衡，代价随之上升。

    [100,1,1,1]：贪心把三个 1 串成链（树高 3），深度受限下最优代价 206。
    """
    weights = [100, 1, 1, 1]
    greedy_height = 3  # 贪心合并 ((100,1),1),1 形态的最大叶深
    assert greedy_height > 2

    result = solve(weights, 2)
    assert result is not None
    assert result.splits == (1, 0, 2)
    assert result.cost == 206  # 100*2 + 1*2 + 1*2 + 1*2
    _, depths = replay(result.splits, len(weights))
    assert max(depths) <= 2


def test_depth_capacity_boundary() -> None:
    """n == 2^D 唯一可行（全满平衡树），n == 2^D + 1 立即不可行。"""
    for d in range(0, 5):
        n = 1 << d
        result = solve([1] * n, d)
        assert result is not None
        if n == 1:
            assert result.splits == ()
            assert result.cost == 0
        else:
            assert max(replay(result.splits, n)[1]) == d
            assert result.cost == d * n  # 所有叶子恰在深度 d

        overflow = solve([1] * (n + 1), d)
        assert overflow is None


def test_depth_capacity_non_power_counts() -> None:
    """n 在容量内但形状受限：D=2 下 n=3 可行，D=1 下 n=3 不可行。"""
    assert solve([1, 1, 1], 2) is not None
    assert solve([1, 1, 1], 1) is None
    assert solve([1, 1], 1) is not None
    assert solve([1, 1], 0) is None


def test_single_symbol_depth_zero_empty_code() -> None:
    """单符号树：深度 0、码字为空串、见证为空、代价 0、唯一。"""
    result = solve([42], 0)
    assert result is not None
    assert result.cost == 0
    assert result.splits == ()
    assert result.count_capped == 1
    assert result.optimality == "unique"
    codes, depths = replay(result.splits, 1)
    assert codes == [""]
    assert depths == [0]

    # D>0 时单符号仍然深度 0（没有理由浪费层数）。
    loose = solve([42], 15)
    assert loose is not None and loose.cost == 0 and loose.splits == ()


def test_multiple_optima_canonical_lex_min() -> None:
    """[1,1,1]：根分界 0 与 1 的两棵树都是最优，必须报告 multiple
    且选择前序分割序列字典序更小者 (0,1)。"""
    result = solve([1, 1, 1], 2)
    assert result is not None
    assert result.cost == 5
    assert result.count_capped == 2
    assert result.optimality == "multiple"
    assert result.splits == (0, 1)
    codes, depths = replay(result.splits, 3)
    assert codes == ["0", "10", "11"]
    assert depths == [1, 2, 2]

    # 另一棵同分价树的见证是 (1,0)；确认它的确同分价（经独立重放）。
    alt_codes, alt_depths = replay((1, 0), 3)
    assert alt_codes == ["00", "01", "1"]
    assert sum(w * d for w, d in zip([1, 1, 1], alt_depths)) == result.cost


def test_unique_report_stays_unique() -> None:
    """权重打破对称时多解坍缩为唯一：重叶必须更浅。"""
    result = solve([1, 1, 10], 2)
    assert result is not None
    assert result.optimality == "unique"
    assert result.splits == (1, 0)  # 重叶子在右侧深度 1
    assert result.cost == 14  # 1*2 + 1*2 + 10*1


def test_max_weights_integer_cost() -> None:
    """权重 10^9、n=40、D=15：精确整数代价，不使用浮点。"""
    weights = [10**9] * 40
    result = solve(weights, 15)
    assert result is not None
    assert isinstance(result.cost, int)
    codes, depths = replay(result.splits, 40)
    assert max(depths) <= 15
    assert result.cost == 10**9 * sum(depths)


def test_max_size_performance() -> None:
    """O(D*n^3) 在 n=40、D=15 上限规模下应在 2 秒内完成。"""
    weights = [(i * 2_654_435_761 + 7) % 10**9 + 1 for i in range(40)]
    start = time.perf_counter()
    result = solve(weights, 15)
    elapsed = time.perf_counter() - start
    assert result is not None
    assert elapsed < 2.0
    replay(result.splits, 40)
