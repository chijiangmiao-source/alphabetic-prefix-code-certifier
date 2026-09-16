"""请求校验：所有错误聚合后按 JSON Pointer 排序、一次返回。

分两层：
1. 结构/类型/取值范围交给 Pydantic（``EncodeRequest``），再把它的错误定位
   ``loc`` 翻译成 RFC 6901 风格的 JSON Pointer（如 ``/symbols/0``）；
2. 只依赖原始 JSON、且结构前提成立时才执行的语义校验手工收集：
   符号必须是单个 ASCII 字符、数组内唯一、两数组等长、禁止未知字段。

两层错误合并去重、按 (pointer, message) 排序后随同一个 422 返回，
绝不在出现错误时继续求解。
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .schemas import EncodeRequest


class ValidationFailure(Exception):
    """请求校验失败；``errors`` 已按 JSON Pointer 排序。"""

    def __init__(self, errors: list[tuple[str, str]]) -> None:
        self.errors = sorted(set(errors))


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _loc_to_pointer(loc: tuple[Any, ...]) -> str:
    if not loc:
        return ""
    return "/" + "/".join(_escape(str(part)) for part in loc)


def parse_body(raw: bytes) -> Any:
    """解析原始请求体；JSON 非法或非对象时以根 Pointer 报错。"""
    if not raw:
        raise ValidationFailure([("", "request body is empty; expected a JSON object")])
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationFailure([("", f"request body is not valid JSON: {exc.msg}")]) from None
    if not isinstance(data, dict):
        raise ValidationFailure(
            [("", "request body must be a JSON object with symbols/weights/max_depth")]
        )
    return data


def validate_request(data: dict[str, Any]) -> EncodeRequest:
    """聚合 Pydantic 结构错误与手工语义错误，全部通过则返回模型实例。"""
    errors: list[tuple[str, str]] = []
    model: EncodeRequest | None = None

    try:
        model = EncodeRequest.model_validate(data)
    except ValidationError as exc:
        for err in exc.errors():
            pointer = _loc_to_pointer(err["loc"])
            errors.append((pointer, err["msg"]))

    symbols = data.get("symbols")
    weights = data.get("weights")

    # 数组等长：仅在两者都是数组时才检查（否则 Pydantic 已报告结构错误）。
    if isinstance(symbols, list) and isinstance(weights, list):
        if len(symbols) != len(weights):
            errors.append(
                (
                    "/weights",
                    f"weights length {len(weights)} must equal symbols length {len(symbols)}",
                )
            )

    # 符号语义：单字符 ASCII、唯一。逐元素检查字符串成员——非字符串元素
    # 交给 Pydantic 报类型错误，但不吞掉其它元素的长度/ASCII/重复错误。
    if isinstance(symbols, list):
        first_seen: dict[str, int] = {}
        for index, symbol in enumerate(symbols):
            if not isinstance(symbol, str):
                continue
            pointer = f"/symbols/{index}"
            if len(symbol) != 1:
                errors.append(
                    (pointer, f"symbol must be a single ASCII character, got {symbol!r}")
                )
                continue
            code = ord(symbol)
            if code > 127:
                errors.append(
                    (pointer, f"symbol must be ASCII (code <= 127), got U+{code:04X}")
                )
            if symbol in first_seen:
                errors.append(
                    (
                        pointer,
                        f"duplicate symbol {symbol!r}; first occurrence at "
                        f"/symbols/{first_seen[symbol]}",
                    )
                )
            else:
                first_seen[symbol] = index

    # 未知字段（extra='forbid' 已由 Pydantic 报错）；这里无需重复。
    if errors:
        raise ValidationFailure(errors)
    assert model is not None
    return model
