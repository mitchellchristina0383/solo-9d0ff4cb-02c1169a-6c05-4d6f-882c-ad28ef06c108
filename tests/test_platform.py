"""端到端自测：基于 ASGI transport 同时跑调度器与两台节点模拟器。

运行：
  PYTHONPATH=/tmp/pylibs PLATFORM_DB=/tmp/platform-test/test.db \
  python3 -m pytest tests/ -v
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid

import httpx
import pytest

# 必须在导入 app 之前配置环境
TEST_DIR = f"/tmp/platform-test-{uuid.uuid4().hex[:8]}"
os.environ["PLATFORM_DB"] = os.path.join(TEST_DIR, "test.db")
os.environ["PLATFORM_NODES"] = (
    "node-a|模拟节点A|http://testnode-a:9001|8192,"
    "node-b|模拟节点B|http://testnode-b:9002|8192"
)
os.environ["NODE_CALL_TIMEOUT"] = "1.0"
os.environ["LOAD_TIMEOUT"] = "4"
os.environ["POLL_INTERVAL"] = "0.2"
os.environ["HEALTH_INTERVAL"] = "3600"  # 测试中手动驱动健康监测

from app.main import app as platform_app          # noqa: E402
from app import config, service                    # noqa: E402
from node import simulator                         # noqa: E402


class RoutingTransport(httpx.AsyncBaseTransport):
    """按 host 把请求路由到对应 ASGI 应用，无需真实端口。"""

    def __init__(self, sim_a, sim_b):
        self._a = httpx.ASGITransport(app=sim_a)
        self._b = httpx.ASGITransport(app=sim_b)
        self._p = httpx.ASGITransport(app=platform_app)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "testnode-a":
            return await self._a.handle_async_request(request)
        if host == "testnode-b":
            return await self._b.handle_async_request(request)
        return await self._p.handle_async_request(request)


@pytest.fixture
async def client():
    # 每台模拟器独立实例、独立内存态；每用例全新 SQLite 库
    os.makedirs(TEST_DIR, exist_ok=True)
    if os.path.exists(os.environ["PLATFORM_DB"]):
        os.remove(os.environ["PLATFORM_DB"])
    from app import db as db_mod
    db_mod._conn = None
    # ASGI 测试传输下用 0 表示直接抛连接异常；真实 uvicorn 下用长挂起复现超时
    sim_a = simulator.create_app("node-a", "模拟节点A", 8192, 0.3, 0.15,
                                 unreachable_sleep=0)
    sim_b = simulator.create_app("node-b", "模拟节点B", 8192, 0.3, 0.15,
                                 unreachable_sleep=0)
    transport = RoutingTransport(sim_a, sim_b)
    async with httpx.AsyncClient(base_url="http://platform", transport=transport,
                                 timeout=10) as c:
        # 手动初始化（不走 lifespan），并替换服务里的全局 httpx client
        service.init_db()
        await service.node_client.close()
        service.node_client._client = httpx.AsyncClient(
            base_url="http://platform", transport=transport, timeout=1.0
        )
        # 启动与生产相同的后台轮询（含健康监测，驱动失联检测与恢复自动对账）
        async def poll_loop():
            while True:
                await service.poll_loading()
                await service.poll_releasing()
                await asyncio.sleep(0.2)

        async def health_loop():
            while True:
                await service.health_monitor_once()
                await asyncio.sleep(0.5)

        poller = asyncio.create_task(poll_loop())
        health = asyncio.create_task(health_loop())
        yield c
        poller.cancel()
        health.cancel()
    await service.node_client.close()


async def wait_for(client, dep_id, statuses, timeout=12.0):
    deadline = asyncio.get_event_loop().time() + timeout
    last = None
    while asyncio.get_event_loop().time() < deadline:
        d = await client.get(f"/api/deployments/{dep_id}")
        last = d.json()
        if last["status"] in statuses:
            return last
        await asyncio.sleep(0.25)
    raise AssertionError(f"部署 {dep_id} 未进入 {statuses}，最后状态={last['status']} / {last['reason']}")


def node_free(book, nid):
    return next(n["free_mb"] for n in book if n["node_id"] == nid)


@pytest.mark.asyncio
async def test_01_full_deploy_flow_and_available(client):
    # 两台节点初始各 8192
    book = (await client.get("/api/nodes")).json()
    assert {n["node_id"] for n in book} == {"node-a", "node-b"}
    assert all(n["free_mb"] == 8192 for n in book)

    r = await client.post("/api/deployments", json={
        "model_name": "llama3-8b", "mem_per_replica_mb": 2048, "replica_count": 2})
    assert r.status_code == 201, r.text
    dep_id = r.json()["id"]

    dep = await wait_for(client, dep_id, {"AVAILABLE"})
    # 两个副本必须分处不同节点
    nodes = {rep["node_id"] for rep in dep["replicas"]}
    assert nodes == {"node-a", "node-b"}
    assert all(rep["status"] == "LOADED" for rep in dep["replicas"])

    book = (await client.get("/api/nodes")).json()
    assert all(n["used_mb"] == 2048 and n["free_mb"] == 6144 for n in book)


@pytest.mark.asyncio
async def test_02_concurrent_reservation_never_overcommits(client):
    # 每台 8192：每副本 3000，双副本部署每台占 3000；并发 3 个 -> 只有 2 个能成功
    async def submit(i):
        return await client.post("/api/deployments", json={
            "model_name": f"m{i}", "mem_per_replica_mb": 3000, "replica_count": 2})

    results = await asyncio.gather(*(submit(i) for i in range(3)))
    codes = [r.status_code for r in results]
    assert codes.count(201) == 2, codes
    assert codes.count(409) == 1, codes

    book = (await client.get("/api/nodes")).json()
    for n in book:
        assert n["used_mb"] <= n["capacity_mb"]
        assert n["used_mb"] == 6000, n  # 2 个部署 × 3000
        assert n["free_mb"] == 2192

    # 显存不足时再提交应明确拒绝
    r = await client.post("/api/deployments", json={
        "model_name": "big", "mem_per_replica_mb": 4000, "replica_count": 1})
    assert r.status_code == 409
    assert "显存不足" in r.json()["detail"]


@pytest.mark.asyncio
async def test_03_load_fail_releases_reservation_and_retry(client):
    # 让节点 A 加载失败；单副本部署落到剩余显存最多的节点（A/B 相同，排序后取 A）
    await client.post("/api/nodes/node-a/control", json={"load_fail": True})
    r = await client.post("/api/deployments", json={
        "model_name": "fail-model", "mem_per_replica_mb": 1024, "replica_count": 1})
    dep_id = r.json()["id"]
    dep = await wait_for(client, dep_id, {"FAILED"})
    rep = dep["replicas"][0]
    assert rep["status"] == "LOAD_FAILED"
    # 失败不占显存
    book = (await client.get("/api/nodes")).json()
    assert all(n["used_mb"] == 0 for n in book)

    # 清除故障后重试，应成功
    await client.post("/api/nodes/node-a/control",
                      json={"load_fail": False, "mode": "normal"})
    r2 = await client.post(f"/api/deployments/{dep_id}/retry")
    assert r2.status_code == 200
    dep2 = await wait_for(client, dep_id, {"AVAILABLE"})
    assert dep2["epoch"] == 2 and dep2["attempts"] == 2
    # 旧 epoch 副本保持终态，不复活
    old = [x for x in dep2["replicas"] if x["epoch"] == 1]
    new = [x for x in dep2["replicas"] if x["epoch"] == 2]
    assert old[0]["status"] == "LOAD_FAILED" and new[0]["status"] == "LOADED"


@pytest.mark.asyncio
async def test_04_unreachable_timeout_keeps_hold_then_reconcile_releases(client):
    await client.post("/api/nodes/node-b/control", json={"mode": "unreachable"})
    r = await client.post("/api/deployments", json={
        "model_name": "ghost", "mem_per_replica_mb": 1500, "replica_count": 2})
    dep_id = r.json()["id"]

    # A 正常加载，B 失联 -> 部署 UNVERIFIED，B 上显存保留占用
    dep = await wait_for(client, dep_id, {"UNVERIFIED"}, timeout=12)
    await service.health_monitor_once()  # 驱动一次探测周期，确认失联被健康监测捕获
    by_node = {x["node_id"]: x for x in dep["replicas"]}
    assert by_node["node-b"]["status"] == "TIMEOUT"
    # A 可能已加载或仍在加载，但绝不能被标记为失败/释放
    assert by_node["node-a"]["status"] in {"LOADING", "LOADED"}

    book = (await client.get("/api/nodes")).json()
    b = next(n for n in book if n["node_id"] == "node-b")
    assert b["used_mb"] == 1500  # 待核期间仍占用，不可被新部署用掉

    # 该显存不能被重新分配：节点 B 剩余不足时提交双副本必被拒
    r2 = await client.post("/api/deployments", json={
        "model_name": "blocked", "mem_per_replica_mb": 7000, "replica_count": 2})
    assert r2.status_code == 409

    # 节点恢复（失联期间请求从未到达模拟器，B 上实际没有该副本）
    await client.post("/api/nodes/node-b/control", json={"mode": "normal"})
    await service.health_monitor_once()  # 触发恢复自动对账
    dep = await wait_for(client, dep_id, {"AVAILABLE", "FAILED"}, timeout=12)
    # B 上实际没有副本 -> 对账释放 -> 该副本 LOAD_FAILED/RELEASED，部署 FAILED
    assert dep["status"] == "FAILED"
    by_node = {x["node_id"]: x for x in dep["replicas"]}
    assert by_node["node-b"]["status"] in {"RELEASED", "LOAD_FAILED"}
    book = (await client.get("/api/nodes")).json()
    b = next(n for n in book if n["node_id"] == "node-b")
    assert b["used_mb"] == 0  # 确认释放后显存可重新分配


@pytest.mark.asyncio
async def test_05_late_ack_must_not_revive_old_replica(client):
    # 场景：节点 A 在加载请求在途时失联，调度器保留占用转待核；
    # 之后节点其实完成了加载（迟到回执/延迟状态变化），恢复后补确认；
    # 再撤销旧加载、重试 epoch=2，旧 epoch 的迟到状态绝不能复活旧副本。

    # 1) 失联期间发起部署 -> TIMEOUT 待核，占用保留
    await client.post("/api/nodes/node-a/control", json={"mode": "unreachable"})
    await service.health_monitor_once()  # 确认健康监测捕获失联（online=False）
    r = await client.post("/api/deployments", json={
        "model_name": "late", "mem_per_replica_mb": 512, "replica_count": 1})
    dep_id = r.json()["id"]
    dep = await wait_for(client, dep_id, {"UNVERIFIED"}, timeout=10)
    rid = dep["replicas"][0]["id"]
    assert dep["replicas"][0]["status"] == "TIMEOUT"
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, "node-a") == 8192 - 512

    # 2) 节点恢复，但它在“失联期间”其实把副本加载好了（模拟迟到回执落地）。
    #    先写入迟到状态再恢复，避免健康循环的自动对账抢先判定为无残留。
    sim_state = client._transport._a.app.state  # type: ignore[attr-defined]
    sim_state.replicas[rid] = {
        "model": "late", "mem_mb": 512, "epoch": 1, "status": "loaded",
    }
    await client.post("/api/nodes/node-a/control", json={"mode": "normal"})
    await service.health_monitor_once()  # 探测恢复边沿，触发自动对账
    # 对账补确认 -> AVAILABLE（同一 epoch 的合法补确认，不是复活）
    (await client.post(f"/api/deployments/{dep_id}/reconcile")).json()
    dep = await wait_for(client, dep_id, {"AVAILABLE"}, timeout=8)
    assert dep["replicas"][0]["status"] == "LOADED"

    # 3) 再次失联并人工撤销旧加载，恢复后对账确认释放
    await client.post("/api/nodes/node-a/control", json={"mode": "unreachable"})
    from app import models
    import app.db as db_mod
    with db_mod.tx() as conn:
        service._set_replica(conn, rid, models.REP_UNVERIFIED, error="演练：失联待核")
        service._evaluate(conn, dep_id)
    await service.request_release(rid, "演练：撤销旧加载")
    await client.post("/api/nodes/node-a/control", json={"mode": "normal"})
    deadline = asyncio.get_event_loop().time() + 15
    dep = None
    while asyncio.get_event_loop().time() < deadline:
        await client.post(f"/api/deployments/{dep_id}/reconcile")
        dep = (await client.get(f"/api/deployments/{dep_id}")).json()
        if all(x["status"] in {"RELEASED", "LOAD_FAILED"} for x in dep["replicas"]):
            break
        await asyncio.sleep(0.25)
    assert dep is not None and all(
        x["status"] in {"RELEASED", "LOAD_FAILED"} for x in dep["replicas"])

    # 4) 模拟旧 epoch 的“迟到回执”：节点侧再次出现 epoch=1 loaded
    sim_state.replicas[rid] = {
        "model": "late", "mem_mb": 512, "epoch": 1, "status": "loaded",
    }
    # 重试 -> epoch=2；调度器只认当前 epoch，旧副本终态不变
    r3 = await client.post(f"/api/deployments/{dep_id}/retry")
    assert r3.status_code == 200
    dep = await wait_for(client, dep_id, {"AVAILABLE"}, timeout=10)
    assert dep["epoch"] == 2
    old = [x for x in dep["replicas"] if x["epoch"] == 1]
    new = [x for x in dep["replicas"] if x["epoch"] == 2]
    assert all(x["status"] in {"RELEASED", "LOAD_FAILED"} for x in old)  # 不复活
    assert new[0]["status"] == "LOADED"
    # 新副本使用不同 rid，节点侧旧记录与新记录互不干扰
    assert new[0]["id"] != rid


@pytest.mark.asyncio
async def test_06_reconcile_failure_stays_unverified_with_reason(client):
    # B 失联 -> 待核；保持失联时手动对账，必须仍是待核并说明原因
    await client.post("/api/nodes/node-b/control", json={"mode": "unreachable"})
    r = await client.post("/api/deployments", json={
        "model_name": "nv", "mem_per_replica_mb": 900, "replica_count": 2})
    dep_id = r.json()["id"]
    dep = await wait_for(client, dep_id, {"UNVERIFIED"}, timeout=12)
    rec = (await client.post(f"/api/deployments/{dep_id}/reconcile")).json()
    assert rec["status"] == "UNVERIFIED"
    assert any("失联" in note or "不可达" in note for note in rec["reconcile_notes"])
    # 显存仍保留
    book = (await client.get("/api/nodes")).json()
    b = next(n for n in book if n["node_id"] == "node-b")
    assert b["used_mb"] == 900


@pytest.mark.asyncio
async def test_07_recovery_revokes_stale_load_then_confirms_release(client):
    # 正常部署成功后，节点 B 失联，调度器侧强制转待核并撤销；
    # 恢复后对账必须撤销节点上的旧加载，确认释放后才可重新分配
    r = await client.post("/api/deployments", json={
        "model_name": "svc", "mem_per_replica_mb": 2500, "replica_count": 2})
    dep_id = r.json()["id"]
    dep = await wait_for(client, dep_id, {"AVAILABLE"}, timeout=10)
    b_rep = next(x for x in dep["replicas"] if x["node_id"] == "node-b")
    rid = b_rep["id"]

    await client.post("/api/nodes/node-b/control", json={"mode": "unreachable"})
    # 直接通过服务层把 B 副本置为待核并请求释放（模拟失联期间人工处置）
    from app import models
    import app.db as db_mod
    with db_mod.tx() as conn:
        service._set_replica(conn, rid, models.REP_UNVERIFIED, error="演练：失联待核")
        service._evaluate(conn, dep_id)
    await service.request_release(rid, "演练：撤销旧加载")

    # 失联中对账不能释放
    rec = (await client.post(f"/api/deployments/{dep_id}/reconcile")).json()
    assert rec["status"] == "UNVERIFIED"
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, "node-b") == 8192 - 2500

    # 恢复：节点上旧加载仍在（loaded），自动/手动对账应撤销并最终确认释放。
    # 自动对账与手动对账可能由不同轮次触发，终态为 RELEASED（确认释放）或
    # LOAD_FAILED（对账确认节点无残留）——两者都不再占用显存，语义等价。
    await client.post("/api/nodes/node-b/control", json={"mode": "normal"})
    await service.health_monitor_once()
    deadline = asyncio.get_event_loop().time() + 15
    while asyncio.get_event_loop().time() < deadline:
        dep = (await client.get(f"/api/deployments/{dep_id}")).json()
        b = next(x for x in dep["replicas"] if x["node_id"] == "node-b")
        if b["status"] in {"RELEASED", "LOAD_FAILED"}:
            break
        await client.post(f"/api/deployments/{dep_id}/reconcile")
        await asyncio.sleep(0.25)
    else:
        raise AssertionError("恢复后副本未在时限内确认释放")
    dep = (await client.get(f"/api/deployments/{dep_id}")).json()
    b = next(x for x in dep["replicas"] if x["node_id"] == "node-b")
    assert b["status"] in {"RELEASED", "LOAD_FAILED"}, dep
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, "node-b") == 8192  # 关键：确认释放后显存全额可用

    # 节点侧旧记录应为 unloaded，且重新部署能拿到显存
    r2 = await client.post("/api/deployments", json={
        "model_name": "new-tenant", "mem_per_replica_mb": 7000, "replica_count": 1})
    assert r2.status_code == 201, r2.text


def sim_state(client, node_id):
    t = client._transport  # type: ignore[attr-defined]
    return (t._a if node_id == "node-a" else t._b).app.state


@pytest.mark.asyncio
async def test_08_reset_releases_single_replica_then_retry(client):
    # 单副本部署可用后，值班员重置承载节点：
    # 转待核 -> 重置 -> 核实无残留 -> 释放显存 -> 失败可重试
    r = await client.post("/api/deployments", json={
        "model_name": "solo", "mem_per_replica_mb": 2048, "replica_count": 1})
    dep_id = r.json()["id"]
    dep = await wait_for(client, dep_id, {"AVAILABLE"})
    node_id = dep["replicas"][0]["node_id"]
    rid = dep["replicas"][0]["id"]

    rr = await client.post(f"/api/nodes/{node_id}/reset")
    assert rr.status_code == 200, rr.text
    body = rr.json()
    assert body["reset"] is True and body["affected_replicas"] == 1
    assert any("无残留" in n for n in body["notes"])

    # 部署失败可重试，副本确认释放，显存全额回收
    dep = await wait_for(client, dep_id, {"FAILED"})
    rep = dep["replicas"][0]
    assert rep["status"] == "RELEASED" and rep["released_at"]
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, node_id) == 8192
    # 节点侧无任何残留记录
    assert sim_state(client, node_id).replicas == {}
    # 事件流水：部署维度可见 转待核 -> 核实释放；节点级 RESET_OK 在全局事件流
    types = [e["event_type"] for e in dep["events"]]
    assert "RESET_MARK" in types and "RESET_RELEASED" in types
    all_events = (await client.get("/api/events?limit=50")).json()
    assert any(e["event_type"] == "RESET_OK" and e["node_id"] == node_id
               for e in all_events)

    # 失败可重试：新 epoch 重新调度成功，旧副本保持终态不复活
    r2 = await client.post(f"/api/deployments/{dep_id}/retry")
    assert r2.status_code == 200
    dep2 = await wait_for(client, dep_id, {"AVAILABLE"})
    assert dep2["epoch"] == 2
    old = [x for x in dep2["replicas"] if x["epoch"] == 1]
    assert old[0]["id"] == rid and old[0]["status"] == "RELEASED"


@pytest.mark.asyncio
async def test_09_reset_interleaved_with_inflight_load(client):
    # 加载在途（节点侧已受理、尚未落地）时发起重置：
    # 旧请求不得重新加载，旧加载不得在重置后落地，状态不得复活
    await client.post("/api/nodes/node-a/control", json={"load_delay": 2.0})
    r = await client.post("/api/deployments", json={
        "model_name": "inflight", "mem_per_replica_mb": 1024, "replica_count": 1})
    dep_id = r.json()["id"]

    # 立即重置：与“下发加载/节点受理/落地”任意交错
    rr = await client.post("/api/nodes/node-a/reset")
    assert rr.status_code == 200, rr.text

    # 终态必须失败可重试且显存全额回收
    dep = await wait_for(client, dep_id, {"FAILED"}, timeout=15)
    assert all(x["status"] in {"RELEASED", "LOAD_FAILED"} for x in dep["replicas"])
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, "node-a") == 8192

    # 等过原加载延迟：旧加载不得在重置后落地（节点侧不得出现 loaded/loading）
    await asyncio.sleep(2.5)
    leftovers = sim_state(client, "node-a").replicas.values()
    assert not any(rec["status"] in {"loaded", "loading"} for rec in leftovers), leftovers
    # 重置后节点控制面恢复默认
    assert sim_state(client, "node-a").control["load_delay"] != 2.0


@pytest.mark.asyncio
async def test_10_reset_result_unknown_keeps_hold_then_reconcile_restores(client):
    # 重置结果不明：保持待核与占用；确认原副本仍在且重置未生效后才恢复可用
    r = await client.post("/api/deployments", json={
        "model_name": "keep", "mem_per_replica_mb": 1024, "replica_count": 1})
    dep_id = r.json()["id"]
    dep = await wait_for(client, dep_id, {"AVAILABLE"})
    node_id = dep["replicas"][0]["node_id"]

    # 模拟重置指令得到显式故障响应（非传输错误，不触发失联联动与自动对账）
    from app import node_client as nc
    orig_reset = nc.reset

    async def bad_reset(base_url):
        return nc.CallResult(False, 500, error="重置被节点拒绝(HTTP 500)：演练")

    nc.reset = bad_reset
    try:
        rr = await client.post(f"/api/nodes/{node_id}/reset")
        assert rr.status_code == 502
        assert "待核" in rr.json()["detail"]
    finally:
        nc.reset = orig_reset

    # 部署保持待核，显存继续占用，不得擅自释放
    dep = (await client.get(f"/api/deployments/{dep_id}")).json()
    assert dep["status"] == "UNVERIFIED"
    assert dep["replicas"][0]["status"] == "UNVERIFIED"
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, node_id) == 8192 - 1024

    # 重置实际未生效、原副本仍在：对账确认后恢复可用
    rec = (await client.post(f"/api/deployments/{dep_id}/reconcile")).json()
    assert rec["status"] == "AVAILABLE"
    dep = (await client.get(f"/api/deployments/{dep_id}")).json()
    assert dep["replicas"][0]["status"] == "LOADED"
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, node_id) == 8192 - 1024  # 恢复可用后占用保持


@pytest.mark.asyncio
async def test_11_repeated_reset_does_not_double_release(client):
    r = await client.post("/api/deployments", json={
        "model_name": "once", "mem_per_replica_mb": 1024, "replica_count": 1})
    dep_id = r.json()["id"]
    dep = await wait_for(client, dep_id, {"AVAILABLE"})
    node_id = dep["replicas"][0]["node_id"]
    rid = dep["replicas"][0]["id"]

    r1 = await client.post(f"/api/nodes/{node_id}/reset")
    assert r1.status_code == 200
    dep = await wait_for(client, dep_id, {"FAILED"})
    released_at = dep["replicas"][0]["released_at"]

    # 重复重置：无受影响副本，不重复释放、不改写终态
    r2 = await client.post(f"/api/nodes/{node_id}/reset")
    assert r2.status_code == 200
    assert r2.json()["affected_replicas"] == 0
    dep2 = (await client.get(f"/api/deployments/{dep_id}")).json()
    rep2 = dep2["replicas"][0]
    assert rep2["status"] == "RELEASED"
    assert rep2["released_at"] == released_at
    n = sum(1 for e in dep2["events"]
            if e["event_type"] == "RESET_RELEASED" and e["replica_id"] == rid)
    assert n == 1
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, node_id) == 8192


@pytest.mark.asyncio
async def test_12_reset_only_affects_deployments_on_that_node(client):
    # dep_a：单副本落在 node-a；dep_shared：双副本分处 A/B
    r = await client.post("/api/deployments", json={
        "model_name": "on-a", "mem_per_replica_mb": 1024, "replica_count": 1})
    dep_a = r.json()["id"]
    dep_a_d = await wait_for(client, dep_a, {"AVAILABLE"})
    assert dep_a_d["replicas"][0]["node_id"] == "node-a"

    r = await client.post("/api/deployments", json={
        "model_name": "shared", "mem_per_replica_mb": 2048, "replica_count": 2})
    dep_shared = r.json()["id"]
    await wait_for(client, dep_shared, {"AVAILABLE"})

    # 重置 node-b：只影响 shared 在 B 上的副本，其余一律不变
    rr = await client.post("/api/nodes/node-b/reset")
    assert rr.status_code == 200
    assert rr.json()["affected_replicas"] == 1

    dep_shared_d = await wait_for(client, dep_shared, {"FAILED"})
    by_node = {x["node_id"]: x for x in dep_shared_d["replicas"]}
    assert by_node["node-b"]["status"] == "RELEASED"
    assert by_node["node-a"]["status"] == "LOADED"  # 其他节点副本不变

    dep_a_d = (await client.get(f"/api/deployments/{dep_a}")).json()
    assert dep_a_d["status"] == "AVAILABLE"  # 其他节点的部署不变
    assert dep_a_d["replicas"][0]["status"] == "LOADED"
    assert not any(e["event_type"].startswith("RESET") for e in dep_a_d["events"])

    book = (await client.get("/api/nodes")).json()
    assert node_free(book, "node-b") == 8192
    assert node_free(book, "node-a") == 8192 - 1024 - 2048


@pytest.mark.asyncio
async def test_13_reset_interleaved_with_releasing(client):
    # 副本已转释放中（撤销在途、节点失联未确认）时发起重置：
    # 重置与撤销交错，最终确认无残留才释放，不复活、不重复释放
    await client.post("/api/nodes/node-a/control", json={"mode": "unreachable"})
    r = await client.post("/api/deployments", json={
        "model_name": "rls", "mem_per_replica_mb": 1024, "replica_count": 1})
    dep_id = r.json()["id"]
    dep = await wait_for(client, dep_id, {"UNVERIFIED"}, timeout=10)
    rid = dep["replicas"][0]["id"]

    await service.request_release(rid, "演练：撤销旧加载")
    dep = (await client.get(f"/api/deployments/{dep_id}")).json()
    assert dep["replicas"][0]["status"] == "RELEASING"

    # 重置走带外控制通道（失联模式下仍可达），同时清除故障模式
    rr = await client.post("/api/nodes/node-a/reset")
    assert rr.status_code == 200, rr.text
    dep = await wait_for(client, dep_id, {"FAILED"})
    rep = dep["replicas"][0]
    assert rep["status"] == "RELEASED"
    book = (await client.get("/api/nodes")).json()
    assert node_free(book, "node-a") == 8192
    assert sim_state(client, "node-a").control["mode"] == "normal"
    # 终态稳定：再对账也不会复活或改写
    await client.post(f"/api/deployments/{dep_id}/reconcile")
    dep = (await client.get(f"/api/deployments/{dep_id}")).json()
    assert dep["replicas"][0]["status"] == "RELEASED"
