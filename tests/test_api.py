"""/api/v1/encode 的端到端测试：成功、不可行、错误聚合排序、约束边界。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def replay_codes(splits: list[int], n: int) -> list[str]:
    positions = iter(splits)
    out: list[str] = []

    def build(i: int, j: int, prefix: str) -> None:
        if i == j:
            out.append((i, prefix))
            return
        k = next(positions)
        assert i <= k < j
        build(i, k, prefix + "0")
        build(k + 1, j, prefix + "1")

    build(0, n - 1, "")
    assert next(positions, None) is None
    return [code for _, code in sorted(out)]


def test_health() -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_encode_basic() -> None:
    body = {"symbols": ["A", "B", "C", "D"], "weights": [2, 2, 1, 2], "max_depth": 15}
    resp = client.post("/api/v1/encode", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "optimal"
    assert data["cost"] == 14
    assert data["optimality"] == "unique"
    assert data["splits"] == [1, 0, 2]
    # 逐符号对齐。
    assert [a["symbol"] for a in data["assignments"]] == list("ABCD")
    assert [a["index"] for a in data["assignments"]] == [0, 1, 2, 3]
    # 每个码字都能由分割见证重放。
    assert replay_codes(data["splits"], 4) == data["codes"]
    assert data["codes"] == ["00", "01", "10", "11"]
    assert data["depths"] == [2, 2, 2, 2]


def test_single_symbol_empty_codeword() -> None:
    resp = client.post(
        "/api/v1/encode",
        json={"symbols": ["Z"], "weights": [1], "max_depth": 0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "optimal"
    assert data["cost"] == 0
    assert data["codes"] == [""]
    assert data["depths"] == [0]
    assert data["splits"] == []
    assert data["assignments"][0]["code"] == ""
    assert data["optimality"] == "unique"


def test_infeasible_no_partial_tree_leak() -> None:
    resp = client.post(
        "/api/v1/encode",
        json={"symbols": list("ABCDE"), "weights": [1] * 5, "max_depth": 2},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "infeasible"
    assert data["capacity"] == 4
    # 不可行结果绝不能泄露半棵树。
    assert "codes" not in data
    assert "depths" not in data
    assert "splits" not in data
    assert "cost" not in data
    assert "assignments" not in data


def test_multiple_optima_flag_and_lex_min_witness() -> None:
    resp = client.post(
        "/api/v1/encode",
        json={"symbols": list("XYZ"), "weights": [1, 1, 1], "max_depth": 2},
    )
    data = resp.json()
    assert resp.status_code == 200
    assert data["status"] == "optimal"
    assert data["optimality"] == "multiple"
    assert data["splits"] == [0, 1]


def test_errors_aggregated_and_sorted_by_pointer() -> None:
    resp = client.post(
        "/api/v1/encode",
        json={
            "symbols": ["A", "A", 3, "é"],
            "weights": [1, 2],
            "max_depth": 16,
        },
    )
    assert resp.status_code == 422
    errors = resp.json()["errors"]
    pointers = [e["pointer"] for e in errors]
    # 一次返回多条且按 JSON Pointer 排序。
    assert pointers == sorted(pointers)
    assert pointers[0] == "/max_depth"
    assert "/weights" in pointers  # 数组不等长
    assert "/symbols/1" in pointers  # 重复
    assert "/symbols/2" in pointers  # 非字符串（Pydantic 类型错误）
    assert "/symbols/3" in pointers  # 非 ASCII
    # 每条错误都带人类可读信息。
    assert all(e["message"] for e in errors)


def test_unknown_field_rejected() -> None:
    resp = client.post(
        "/api/v1/encode",
        json={"symbols": ["A"], "weights": [1], "max_depth": 0, "extra": 1},
    )
    assert resp.status_code == 422
    pointers = [e["pointer"] for e in resp.json()["errors"]]
    assert "/extra" in pointers


def test_weight_bounds_and_size_bounds() -> None:
    base = {"symbols": ["A"], "weights": [0], "max_depth": 0}
    assert client.post("/api/v1/encode", json=base).status_code == 422
    base["weights"] = [10**9 + 1]
    assert client.post("/api/v1/encode", json=base).status_code == 422

    too_many = {
        "symbols": [chr(65 + (i % 26)) + str(i) for i in range(41)],
        "weights": [1] * 41,
        "max_depth": 15,
    }
    # 符号还必须是单字符；这里主要验证数量上限，改用合法唯一 ASCII。
    too_many["symbols"] = [chr(i) for i in range(33, 33 + 41)]
    r = client.post("/api/v1/encode", json=too_many)
    assert r.status_code == 422


def test_malformed_json() -> None:
    resp = client.post("/api/v1/encode", content=b"{not json", headers={})
    assert resp.status_code == 422
    errors = resp.json()["errors"]
    assert errors[0]["pointer"] == ""
    assert "JSON" in errors[0]["message"]


def test_non_utf8_body_is_request_format_error_not_500() -> None:
    """无法按 UTF-8 解码的请求体必须返回 422 请求格式错误，而非 500。"""
    resp = client.post(
        "/api/v1/encode",
        content=b"\xff\xfe{not utf8",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    errors = resp.json()["errors"]
    assert len(errors) == 1
    assert errors[0]["pointer"] == ""
    assert "UTF-8" in errors[0]["message"]


def test_empty_body() -> None:
    resp = client.post("/api/v1/encode", content=b"")
    assert resp.status_code == 422
    assert resp.json()["errors"][0]["pointer"] == ""


def test_non_object_body() -> None:
    resp = client.post("/api/v1/encode", json=[1, 2, 3])
    # TestClient 会把 list 作为 JSON 发送；服务端应拒绝非对象。
    assert resp.status_code == 422
    assert resp.json()["errors"][0]["pointer"] == ""


@pytest.mark.parametrize(
    "weights,d,cost",
    [
        ([100, 1, 1, 1], 2, 206),
        ([1, 1, 1, 1], 2, 8),
        ([1] * 8, 3, 24),
    ],
)
def test_known_costs(weights: list[int], d: int, cost: int) -> None:
    body = {
        "symbols": [chr(65 + i) for i in range(len(weights))],
        "weights": weights,
        "max_depth": d,
    }
    data = client.post("/api/v1/encode", json=body).json()
    assert data["status"] == "optimal"
    assert data["cost"] == cost
    assert max(data["depths"]) <= d
    assert replay_codes(data["splits"], len(weights)) == data["codes"]
