"""把求解结果组装为响应模型。

码字与叶深不直接信任 DP 内部状态，而是用 ``solver.replay`` 仅凭分割见证
重放整棵树得到——这保证了「每个码字都能由分割见证重放」的可验证性，
也顺便校验见证合法性；再独立核验一次深度上限与代价总和。
"""

from __future__ import annotations

from .schemas import Assignment, EncodeRequest, InfeasibleResponse, OptimalResponse
from .solver import replay, solve


def build_response(request: EncodeRequest) -> OptimalResponse | InfeasibleResponse:
    n = len(request.symbols)
    result = solve(request.weights, request.max_depth)

    if result is None:
        # 不可行：只返回状态/原因/容量，绝不带任何部分树信息。
        return InfeasibleResponse(
            status="infeasible",
            reason=(
                f"no alphabetic prefix tree exists with {n} leaves at depth "
                f"<= {request.max_depth}; a binary tree of depth {request.max_depth} "
                f"holds at most 2**{request.max_depth} = {1 << request.max_depth} leaves"
            ),
            capacity=1 << request.max_depth,
        )

    if len(result.splits) != n - 1:
        # 内部一致性断言：满二叉树的内部节点数恰为 n-1。
        raise RuntimeError("solver produced an internal-node witness of wrong length")

    codes, depths = replay(result.splits, n)

    # 独立复核（而非再次信任 DP）：深度、代价、前缀-free、叶序区间。
    if any(d > request.max_depth for d in depths):
        raise RuntimeError("replayed tree violates the depth bound")
    verified_cost = sum(w * d for w, d in zip(request.weights, depths))
    if verified_cost != result.cost:
        raise RuntimeError("replayed cost disagrees with the DP optimum")
    _assert_prefix_free_in_order(codes)

    assignments = [
        Assignment(index=i, symbol=request.symbols[i], code=codes[i], depth=depths[i])
        for i in range(n)
    ]
    return OptimalResponse(
        status="optimal",
        cost=result.cost,
        optimality=result.optimality,
        codes=codes,
        depths=depths,
        splits=list(result.splits),
        assignments=assignments,
    )


def _assert_prefix_free_in_order(codes: list[str]) -> None:
    """码字必须两两前缀无关，且按字典序与输入叶序一致（左 0 右 1）。"""
    ordered = sorted(range(len(codes)), key=lambda i: codes[i])
    if ordered != list(range(len(codes))):
        raise RuntimeError("replayed codes are not ordered consistently with leaf order")
    for a, b in zip(codes, codes[1:]):
        if b.startswith(a):
            raise RuntimeError("replayed codes violate the prefix-free property")
