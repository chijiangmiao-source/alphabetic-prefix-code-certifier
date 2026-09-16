"""小规模全树穷举（Catalan 枚举）复核求解器。

对每个实例枚举所有「叶子有序、内部节点二叉」的树，检查深度上限，
独立计算：最优代价、达到最优的不同前序分割序列数量（精确）、
其中字典序最小的分割序列。然后与 DP 的封顶计数和 canonical 见证逐一比对。
"""

from __future__ import annotations

import itertools
import random

import pytest

from app.solver import COUNT_CAP, replay, solve


def enumerate_trees(i: int, j: int, depth: int, max_depth: int):
    """生成 ((split_signature), depths) 的所有可行树；叶 i..j 当前深度 depth。"""
    if depth > max_depth:
        return
    if i == j:
        depths = [0] * 0
        yield (), {i: depth}
        return
    for k in range(i, j):
        for left_sig, left_depths in enumerate_trees(i, k, depth + 1, max_depth):
            for right_sig, right_depths in enumerate_trees(
                k + 1, j, depth + 1, max_depth
            ):
                merged = {**left_depths, **right_depths}
                yield (k,) + left_sig + right_sig, merged


def brute_opt(weights: list[int], max_depth: int):
    """返回 (min_cost, exact_count, lex_min_signature) 或 None（不可行）。"""
    n = len(weights)
    candidates = []
    for sig, depths in enumerate_trees(0, n - 1, 0, max_depth):
        cost = sum(weights[i] * depths[i] for i in range(n))
        candidates.append((cost, sig))
    if not candidates:
        return None
    minimum = min(cost for cost, _ in candidates)
    optimal_signatures = sorted({sig for cost, sig in candidates if cost == minimum})
    return minimum, len(optimal_signatures), optimal_signatures[0]


@pytest.mark.parametrize("seed", range(120))
def test_dp_matches_full_enumeration_random(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(1, 7)
    max_depth = rng.randint(0, 4)
    weights = [rng.randint(1, 9) for _ in range(n)]

    expected = brute_opt(weights, max_depth)
    result = solve(weights, max_depth)

    if expected is None:
        assert result is None
        return

    min_cost, exact_count, lex_min_sig = expected
    assert result is not None
    assert result.cost == min_cost
    assert result.splits == lex_min_sig
    assert result.count_capped == min(COUNT_CAP, exact_count)
    assert result.optimality == ("unique" if exact_count == 1 else "multiple")


def test_all_equal_weights_small() -> None:
    """等权重是歧义高发场景：对 n<=7、D<=3 做地毯式穷举比对。"""
    for n in range(1, 8):
        for d in range(0, 4):
            weights = [1] * n
            expected = brute_opt(weights, d)
            result = solve(weights, d)
            if expected is None:
                assert result is None
            else:
                assert result.cost == expected[0]
                assert result.splits == expected[2]
                assert result.count_capped == min(2, expected[1])


def test_weight_vectors_cartesian() -> None:
    """小笛卡尔积：覆盖各种平局打破方式，代价/计数/见证必须全部吻合穷举。"""
    for n in (2, 3, 4):
        for weights in itertools.product(range(1, 4), repeat=n):
            for d in (n - 1, n):  # 一个紧、一个松的深度预算
                expected = brute_opt(list(weights), d)
                result = solve(list(weights), d)
                assert result is not None and expected is not None
                assert result.cost == expected[0]
                assert result.splits == expected[2]
                assert result.count_capped == min(2, expected[1])


def test_replay_reconstructs_every_code() -> None:
    """任何分割见证都必须能重放出全部码字，且码字前缀无关、按叶序排列。"""
    rng = random.Random(42)
    for _ in range(200):
        n = rng.randint(1, 12)
        d = rng.randint(max(1, (n - 1).bit_length()), 8)
        weights = [rng.randint(1, 1_000) for _ in range(n)]
        result = solve(weights, d)
        assert result is not None
        codes, depths = replay(result.splits, n)

        assert len(result.splits) == n - 1
        assert len(codes) == n and all(c == "0" * 0 or set(c) <= {"0", "1"} for c in codes)
        # 深度不超过 D。
        assert max(depths, default=0) <= d
        # 左 0 右 1 ⇒ 码字字典序必须与叶序严格一致。
        assert codes == sorted(codes)
        # 前缀无关。
        for earlier, later in zip(codes, codes[1:]):
            assert not later.startswith(earlier)
        # Kraft 和 == 1（满二叉树）。
        assert sum(2 ** -dep for dep in depths) == 1.0
        # 代价与重放深度一致。
        assert sum(w * dep for w, dep in zip(weights, depths)) == result.cost


def test_replay_rejects_corrupt_witness() -> None:
    with pytest.raises(ValueError):
        replay((5,), 4)            # 越界分界
    with pytest.raises(ValueError):
        replay((0, 0), 2)          # 合法重放后仍有残余分界
    with pytest.raises(ValueError):
        replay((0,), 3)            # 分界不足
    with pytest.raises(ValueError):
        replay((), 2)              # 完全缺失分界
