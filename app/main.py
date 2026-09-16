"""FastAPI 入口。

POST /api/v1/encode
    请求体：{"symbols": ["A", ...], "weights": [3, ...], "max_depth": 3}
    200 ：{"status": "optimal", ...} 或 {"status": "infeasible", ...}
    422 ：{"errors": [{"pointer": "/weights/1", "message": "..."}, ...]}
          —— 所有错误按 JSON Pointer 排序后一次返回。
GET  /health ：存活探针。
"""

from __future__ import annotations

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from .schemas import ErrorItem, ErrorResponse, InfeasibleResponse, OptimalResponse
from .service import build_response
from .validation import ValidationFailure, parse_body, validate_request

app = FastAPI(
    title="Depth-Limited Alphabetic Prefix Code Service",
    version="1.0.0",
    description=(
        "为按数组顺序固定的唯一 ASCII 符号求深度受限、保叶序的最优二进制前缀树。"
    ),
)


@app.exception_handler(ValidationFailure)
async def validation_failure_handler(_: Request, exc: ValidationFailure) -> JSONResponse:
    payload = ErrorResponse(
        errors=[ErrorItem(pointer=p, message=m) for p, m in exc.errors]
    ).model_dump()
    return JSONResponse(status_code=422, content=payload)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/api/v1/encode",
    response_model=OptimalResponse | InfeasibleResponse,
    responses={422: {"model": ErrorResponse}},
)
async def encode(request: Request) -> Response:
    raw = await request.body()
    data = parse_body(raw)
    model = validate_request(data)
    result = build_response(model)
    return JSONResponse(result.model_dump())
