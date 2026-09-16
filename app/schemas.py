"""Pydantic I/O 模型。

请求与响应的结构、取值范围均由模型声明（同时体现在 /docs 的 OpenAPI 中）；
跨字段语义校验（ASCII、唯一性、数组对齐等）见 ``app.validation``，
那里需要把多条错误按 JSON Pointer 聚合后一次返回。
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

Weight = Annotated[StrictInt, Field(ge=1, le=10**9)]
Depth = Annotated[StrictInt, Field(ge=0, le=15)]

N_MIN, N_MAX = 1, 40


class EncodeRequest(BaseModel):
    """编码请求：符号数组与权重数组按相同下标对齐，顺序即叶序。"""

    model_config = ConfigDict(extra="forbid")

    symbols: list[str] = Field(min_length=N_MIN, max_length=N_MAX)
    weights: list[Weight] = Field(min_length=N_MIN, max_length=N_MAX)
    max_depth: Depth


class Assignment(BaseModel):
    """单个符号的编码结果。"""

    index: int
    symbol: str
    code: str
    depth: int


class OptimalResponse(BaseModel):
    """可行时的响应：代价、逐符号码字/叶深、前序分割见证与最优性标记。"""

    status: Literal["optimal"]
    cost: int
    optimality: Literal["unique", "multiple"]
    codes: list[str]
    depths: list[int]
    splits: list[int]
    assignments: list[Assignment]


class InfeasibleResponse(BaseModel):
    """不可行时的响应。

    只包含状态、原因与容量上界，绝不携带任何码表、深度或分割等部分树信息。
    """

    status: Literal["infeasible"]
    reason: str
    capacity: StrictInt


class ErrorItem(BaseModel):
    pointer: str
    message: str


class ErrorResponse(BaseModel):
    errors: list[ErrorItem]
