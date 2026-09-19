"""API 行为测试：租约独占/过期、幂等上报、提交与状态查询。"""
import os
import tempfile
import threading

# 必须在导入 app 之前指定测试数据库
os.environ["UPLOAD_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ["DEFAULT_LEASE_TTL_SECONDS"] = "300"

from fastapi.testclient import TestClient  # noqa: E402

from app.db import get_conn  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


def make_session(total_chunks=3, file_digest="f" * 64):
    resp = client.post("/sessions", json={
        "total_chunks": total_chunks,
        "file_digest": file_digest,
        "digest_algorithm": "sha256",
    })
    assert resp.status_code == 201, resp.text
    return resp.json()


def lease(session_id, index, ttl=None):
    body = {} if ttl is None else {"lease_ttl_seconds": ttl}
    return client.post(f"/sessions/{session_id}/chunks/{index}/lease", json=body)


def complete(session_id, index, token, digest):
    return client.post(
        f"/sessions/{session_id}/chunks/{index}/complete",
        json={"lease_token": token, "chunk_digest": digest},
    )


def test_create_session_and_status():
    s = make_session(total_chunks=4)
    assert s["status"] == "active"
    assert s["total_chunks"] == 4

    st = client.get(f"/sessions/{s['id']}").json()
    assert st["completed_chunks"] == 0
    assert st["missing_chunks"] == [0, 1, 2, 3]
    assert [c["status"] for c in st["chunks"]] == ["pending"] * 4


def test_lease_is_exclusive_until_expiry():
    s = make_session()
    r1 = lease(s["id"], 0)
    assert r1.status_code == 200
    token1 = r1.json()["lease_token"]
    assert token1

    # 同一分片在租约有效期内不可再次领取
    r2 = lease(s["id"], 0)
    assert r2.status_code == 409

    # 手动让租约过期后，可被重新领取
    get_conn().execute(
        "UPDATE chunks SET lease_expires_at = '2000-01-01T00:00:00+00:00'"
        " WHERE session_id = ? AND chunk_index = 0",
        (s["id"],),
    )
    get_conn().commit()
    r3 = lease(s["id"], 0)
    assert r3.status_code == 200
    assert r3.json()["lease_token"] != token1


def test_complete_requires_valid_lease_and_token():
    s = make_session()
    # 未领取租约直接上报
    assert complete(s["id"], 0, "no-such-token", "a" * 64).status_code == 409

    token = lease(s["id"], 0).json()["lease_token"]
    # 令牌不匹配
    assert complete(s["id"], 0, "wrong-token", "a" * 64).status_code == 409
    # 正确令牌
    assert complete(s["id"], 0, token, "a" * 64).status_code == 200


def test_expired_lease_cannot_complete():
    s = make_session()
    token = lease(s["id"], 0).json()["lease_token"]
    get_conn().execute(
        "UPDATE chunks SET lease_expires_at = '2000-01-01T00:00:00+00:00'"
        " WHERE session_id = ? AND chunk_index = 0",
        (s["id"],),
    )
    get_conn().commit()
    assert complete(s["id"], 0, token, "a" * 64).status_code == 409


def test_complete_is_idempotent():
    s = make_session()
    token = lease(s["id"], 1).json()["lease_token"]
    digest = "b" * 64

    r1 = complete(s["id"], 1, token, digest)
    assert r1.status_code == 200
    assert r1.json()["idempotent_replay"] is False

    # 相同令牌 + 相同摘要：幂等成功，completed_at 不变
    r2 = complete(s["id"], 1, token, digest)
    assert r2.status_code == 200
    assert r2.json()["idempotent_replay"] is True
    assert r2.json()["completed_at"] == r1.json()["completed_at"]

    # 相同令牌但摘要不同：冲突
    assert complete(s["id"], 1, token, "c" * 64).status_code == 409
    # 已完成分片不可再领取租约
    assert lease(s["id"], 1).status_code == 409


def test_finalize_blocked_until_all_chunks_done():
    s = make_session(total_chunks=3)
    token = lease(s["id"], 0).json()["lease_token"]
    assert complete(s["id"], 0, token, "0" * 64).status_code == 200

    r = client.post(f"/sessions/{s['id']}/finalize")
    assert r.status_code == 409
    assert r.json()["missing_chunks"] == [1, 2]

    for i in (1, 2):
        t = lease(s["id"], i).json()["lease_token"]
        assert complete(s["id"], i, t, str(i) * 64).status_code == 200

    ok = client.post(f"/sessions/{s['id']}/finalize")
    assert ok.status_code == 200
    body = ok.json()
    assert body["status"] == "finalized"
    assert body["file_digest"] == "f" * 64
    # 清单按分片序号升序汇总
    assert [m["chunk_index"] for m in body["manifest"]] == [0, 1, 2]
    assert [m["chunk_digest"] for m in body["manifest"]] == [str(i) * 64 for i in range(3)]

    # 重复提交幂等
    again = client.post(f"/sessions/{s['id']}/finalize")
    assert again.status_code == 200
    assert again.json()["manifest"] == body["manifest"]

    # 已提交会话不可再领取租约
    assert lease(s["id"], 0).status_code == 409

    st = client.get(f"/sessions/{s['id']}").json()
    assert st["status"] == "finalized"
    assert st["missing_chunks"] == []
    assert st["completed_chunks"] == 3


def test_unknown_session_and_chunk():
    assert client.get("/sessions/nope").status_code == 404
    s = make_session()
    assert lease(s["id"], 99).status_code == 404
    assert complete(s["id"], 99, "t", "d").status_code == 404


def test_validation():
    assert client.post("/sessions", json={"total_chunks": 0, "file_digest": "x"}).status_code == 422
    assert client.post("/sessions", json={"total_chunks": 1}).status_code == 422


# ---------------------------------------------------------------- 依赖调度

def make_session_with_deps(total_chunks, dependencies):
    resp = client.post("/sessions", json={
        "total_chunks": total_chunks,
        "file_digest": "f" * 64,
        "digest_algorithm": "sha256",
        "dependencies": {str(k): v for k, v in dependencies.items()},
    })
    assert resp.status_code == 201, resp.text
    return resp.json()


def ready(session_id):
    resp = client.get(f"/sessions/{session_id}/ready")
    assert resp.status_code == 200, resp.text
    return resp.json()


def expire_lease(session_id, index):
    get_conn().execute(
        "UPDATE chunks SET lease_expires_at = '2000-01-01T00:00:00+00:00'"
        " WHERE session_id = ? AND chunk_index = ?",
        (session_id, index),
    )
    get_conn().commit()


def create_session_raw(payload):
    return client.post("/sessions", json={"file_digest": "f" * 64, **payload})


def test_dependencies_rejected_when_invalid():
    # 键越界
    assert create_session_raw({"total_chunks": 3, "dependencies": {"3": [0]}}).status_code == 422
    assert create_session_raw({"total_chunks": 3, "dependencies": {"-1": [0]}}).status_code == 422
    # 值越界
    assert create_session_raw({"total_chunks": 3, "dependencies": {"1": [3]}}).status_code == 422
    assert create_session_raw({"total_chunks": 3, "dependencies": {"1": [-2]}}).status_code == 422
    # 自依赖
    assert create_session_raw({"total_chunks": 3, "dependencies": {"1": [1]}}).status_code == 422
    # 环：二元环与三元环
    assert create_session_raw(
        {"total_chunks": 3, "dependencies": {"0": [1], "1": [0]}}).status_code == 422
    assert create_session_raw(
        {"total_chunks": 3, "dependencies": {"0": [1], "1": [2], "2": [0]}}).status_code == 422
    # 非数值键
    assert create_session_raw({"total_chunks": 3, "dependencies": {"x": [0]}}).status_code == 422
    # 合法依赖（含重复值，服务端去重）可创建
    assert create_session_raw(
        {"total_chunks": 3, "dependencies": {"2": [0, 1, 1]}}).status_code == 201


def test_lease_blocked_until_dependencies_complete():
    s = make_session_with_deps(3, {2: [0, 1]})

    # 前置全部未完成：409 + 按序缺失前置
    r = lease(s["id"], 2)
    assert r.status_code == 409
    assert r.json()["missing_dependencies"] == [0, 1]

    # 完成一个前置后仍被阻塞，缺失列表随之缩小
    t0 = lease(s["id"], 0).json()["lease_token"]
    assert complete(s["id"], 0, t0, "0" * 64).status_code == 200
    r = lease(s["id"], 2)
    assert r.status_code == 409
    assert r.json()["missing_dependencies"] == [1]

    # 前置全部完成后无需额外操作即可领取
    t1 = lease(s["id"], 1).json()["lease_token"]
    assert complete(s["id"], 1, t1, "1" * 64).status_code == 200
    assert lease(s["id"], 2).status_code == 200

    # 无依赖的分片不受影响
    s2 = make_session()
    assert lease(s2["id"], 2).status_code == 200


def test_status_exposes_dependencies():
    s = make_session_with_deps(3, {2: [0, 1]})
    st = client.get(f"/sessions/{s['id']}").json()
    assert [c["dependencies"] for c in st["chunks"]] == [[], [], [0, 1]]

    # 未提供依赖的会话：每个分片 dependencies 为空
    st2 = client.get(f"/sessions/{make_session()['id']}").json()
    assert [c["dependencies"] for c in st2["chunks"]] == [[], [], []]


def test_ready_endpoint_lists_ready_and_blocked():
    s = make_session_with_deps(4, {2: [0], 3: [1, 2]})

    # 初始：0、1 可领取；2 等 0，3 等 1 和 2
    r = ready(s["id"])
    assert r["ready_chunks"] == [0, 1]
    assert r["blocked_chunks"] == [
        {"chunk_index": 2, "missing_dependencies": [0]},
        {"chunk_index": 3, "missing_dependencies": [1, 2]},
    ]

    # 持有有效租约的分片不再出现在 ready 中
    t0 = lease(s["id"], 0).json()["lease_token"]
    assert ready(s["id"])["ready_chunks"] == [1]

    # 完成 0 后 2 解锁
    assert complete(s["id"], 0, t0, "0" * 64).status_code == 200
    r = ready(s["id"])
    assert r["ready_chunks"] == [1, 2]
    assert r["blocked_chunks"] == [{"chunk_index": 3, "missing_dependencies": [1, 2]}]

    # 链式推进：完成 1、2 后 3 解锁
    t1 = lease(s["id"], 1).json()["lease_token"]
    assert complete(s["id"], 1, t1, "1" * 64).status_code == 200
    t2 = lease(s["id"], 2).json()["lease_token"]
    assert complete(s["id"], 2, t2, "2" * 64).status_code == 200
    r = ready(s["id"])
    assert r["ready_chunks"] == [3]
    assert r["blocked_chunks"] == []


def test_ready_endpoint_treats_expired_lease_as_claimable():
    s = make_session_with_deps(2, {1: [0]})

    lease(s["id"], 0)
    assert ready(s["id"])["ready_chunks"] == []

    # 租约过期后恢复为可领取
    expire_lease(s["id"], 0)
    assert ready(s["id"])["ready_chunks"] == [0]

    # 无依赖会话：全部 pending 分片均可领取，无阻塞
    s2 = make_session()
    r = ready(s2["id"])
    assert r["ready_chunks"] == [0, 1, 2]
    assert r["blocked_chunks"] == []


def test_ready_unknown_session():
    assert client.get("/sessions/nope/ready").status_code == 404


def test_finalize_with_dependencies():
    s = make_session_with_deps(2, {1: [0]})
    # 被依赖阻塞时无法完成后续分片，提交自然列出缺失
    r = client.post(f"/sessions/{s['id']}/finalize")
    assert r.status_code == 409
    assert r.json()["missing_chunks"] == [0, 1]

    for i in (0, 1):
        t = lease(s["id"], i).json()["lease_token"]
        assert complete(s["id"], i, t, str(i) * 64).status_code == 200
    assert client.post(f"/sessions/{s['id']}/finalize").status_code == 200


# ---------------------------------------------------------------- 会话级并发上限

def make_capped_session(total_chunks, max_active_leases, dependencies=None):
    payload = {
        "total_chunks": total_chunks,
        "file_digest": "f" * 64,
        "digest_algorithm": "sha256",
        "max_active_leases": max_active_leases,
    }
    if dependencies is not None:
        payload["dependencies"] = {str(k): v for k, v in dependencies.items()}
    resp = client.post("/sessions", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


def batch(session_id, count, ttl=None):
    body = {"count": count}
    if ttl is not None:
        body["lease_ttl_seconds"] = ttl
    return client.post(f"/sessions/{session_id}/leases/batch", json=body)


def test_max_active_leases_validation():
    assert client.post("/sessions", json={
        "total_chunks": 3, "file_digest": "f" * 64, "max_active_leases": 0,
    }).status_code == 422
    assert client.post("/sessions", json={
        "total_chunks": 3, "file_digest": "f" * 64, "max_active_leases": -1,
    }).status_code == 422
    # 不设置或为 null：不限制
    assert make_session()["max_active_leases"] is None
    assert client.post("/sessions", json={
        "total_chunks": 3, "file_digest": "f" * 64, "max_active_leases": None,
    }).json()["max_active_leases"] is None


def test_status_exposes_cap_and_active_count():
    s = make_capped_session(5, max_active_leases=2)
    st = client.get(f"/sessions/{s['id']}").json()
    assert st["max_active_leases"] == 2
    assert st["active_leases"] == 0

    batch(s["id"], count=2)
    st = client.get(f"/sessions/{s['id']}").json()
    assert st["active_leases"] == 2

    # 过期租约不计入有效租约数
    for i in (0, 1):
        expire_lease(s["id"], i)
    st = client.get(f"/sessions/{s['id']}").json()
    assert st["active_leases"] == 0

    # 完成分片释放槽位
    t0 = lease(s["id"], 0).json()["lease_token"]
    assert complete(s["id"], 0, t0, "0" * 64).status_code == 200
    st = client.get(f"/sessions/{s['id']}").json()
    assert st["active_leases"] == 0


def test_batch_grants_in_index_order_capped_by_count():
    s = make_session()
    r = batch(s["id"], count=2, ttl=120)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["granted_count"] == 2
    assert [item["chunk_index"] for item in body["leases"]] == [0, 1]
    assert all(item["lease_token"] for item in body["leases"])
    # 同一批量内所有租约同时到期
    expires = {item["expires_at"] for item in body["leases"]}
    assert len(expires) == 1
    assert body["active"] == 2
    assert body["remaining_slots"] is None  # 未设置并发上限
    assert body["blocked_chunks"] == []
    assert body["next_expiry"] is not None


def test_batch_respects_max_active_leases():
    s = make_capped_session(5, max_active_leases=3)

    r1 = batch(s["id"], count=2)
    assert [i["chunk_index"] for i in r1.json()["leases"]] == [0, 1]
    assert r1.json()["active"] == 2
    assert r1.json()["remaining_slots"] == 1

    # 仅剩 1 个槽位：请求 3 个也只发放 1 个
    r2 = batch(s["id"], count=3)
    body = r2.json()
    assert body["granted_count"] == 1
    assert [i["chunk_index"] for i in body["leases"]] == [2]
    assert body["active"] == 3
    assert body["remaining_slots"] == 0

    # 槽位耗尽：返回空列表与汇总
    r3 = batch(s["id"], count=2)
    body = r3.json()
    assert body["granted_count"] == 0
    assert body["leases"] == []
    assert body["active"] == 3
    assert body["remaining_slots"] == 0
    assert body["next_expiry"] is not None

    # 单片接口同样受会话上限约束
    assert lease(s["id"], 3).status_code == 409

    # 完成一个分片后槽位释放，可继续领取
    t2 = next(i["lease_token"] for i in r2.json()["leases"] if i["chunk_index"] == 2)
    assert complete(s["id"], 2, t2, "2" * 64).status_code == 200
    r4 = batch(s["id"], count=2)
    assert [i["chunk_index"] for i in r4.json()["leases"]] == [3]
    assert r4.json()["active"] == 3


def test_batch_expired_leases_free_slots():
    s = make_capped_session(3, max_active_leases=2)
    batch(s["id"], count=2)
    assert batch(s["id"], count=2).json()["granted_count"] == 0

    # 过期一个租约：既释放槽位，该分片本身也重新可领取；
    # 分片 1 仍持有效租约，故只有 1 个槽位，领回分片 0
    expire_lease(s["id"], 0)
    body = batch(s["id"], count=2).json()
    assert [i["chunk_index"] for i in body["leases"]] == [0]
    assert body["active"] == 2
    assert body["remaining_slots"] == 0

    # 分片 1 也过期后：分片 0 仍持有效新约，仅剩 1 槽，按序号领回分片 1
    expire_lease(s["id"], 1)
    body = batch(s["id"], count=2).json()
    assert [i["chunk_index"] for i in body["leases"]] == [1]
    assert body["active"] == 2


def test_batch_skips_held_chunks():
    s = make_capped_session(4, max_active_leases=4)
    # 单片接口先占用 1
    lease(s["id"], 1)
    body = batch(s["id"], count=4).json()
    assert [i["chunk_index"] for i in body["leases"]] == [0, 2, 3]
    assert body["active"] == 4


def test_batch_respects_dependencies_and_reports_blocked():
    s = make_capped_session(4, max_active_leases=4, dependencies={2: [0], 3: [1, 2]})

    body = batch(s["id"], count=4).json()
    # 仅 0、1 的前置已完成（无依赖）；2、3 被阻塞
    assert [i["chunk_index"] for i in body["leases"]] == [0, 1]
    assert body["active"] == 2
    assert body["blocked_chunks"] == [
        {"chunk_index": 2, "missing_dependencies": [0]},
        {"chunk_index": 3, "missing_dependencies": [1, 2]},
    ]

    # 完成 0 后，2 解除阻塞，批量领取得到 2
    t0 = next(i["lease_token"] for i in body["leases"] if i["chunk_index"] == 0)
    assert complete(s["id"], 0, t0, "0" * 64).status_code == 200
    body2 = batch(s["id"], count=4).json()
    assert [i["chunk_index"] for i in body2["leases"]] == [2]
    assert body2["blocked_chunks"] == [
        {"chunk_index": 3, "missing_dependencies": [1, 2]},
    ]


def test_batch_empty_when_nothing_claimable():
    s = make_capped_session(2, max_active_leases=2)
    batch(s["id"], count=2)
    body = batch(s["id"], count=2).json()
    assert body["leases"] == []
    assert body["granted_count"] == 0
    assert body["active"] == 2
    assert body["next_expiry"] is not None

    # 全部完成后：无有效租约、无阻塞
    for i in (0, 1):
        # 从数据库直接取当前令牌完成
        row = get_conn().execute(
            "SELECT lease_token FROM chunks WHERE session_id = ? AND chunk_index = ?",
            (s["id"], i),
        ).fetchone()
        assert complete(s["id"], i, row["lease_token"], str(i) * 64).status_code == 200
    body = batch(s["id"], count=2).json()
    assert body["leases"] == []
    assert body["active"] == 0
    assert body["remaining_slots"] == 2
    assert body["blocked_chunks"] == []
    assert body["next_expiry"] is None


def test_batch_ttl_is_clamped_to_max():
    s = make_capped_session(2, max_active_leases=2)
    r = batch(s["id"], count=1, ttl=10 ** 9)
    assert r.status_code == 200
    expires_at = r.json()["leases"][0]["expires_at"]
    from datetime import datetime
    delta = datetime.fromisoformat(expires_at) - utcnow_local()
    # 上限 3600 秒（测试环境 MAX_LEASE_TTL_SECONDS 默认）
    assert delta.total_seconds() <= 3600 + 5


def utcnow_local():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


def test_batch_validation_and_errors():
    # count 非法
    s = make_capped_session(2, max_active_leases=2)
    assert batch(s["id"], count=0).status_code == 422
    # 未知会话
    assert batch("nope", count=1).status_code == 404
    # 已提交会话拒绝批量领取
    s2 = make_session(total_chunks=1)
    t = lease(s2["id"], 0).json()["lease_token"]
    assert complete(s2["id"], 0, t, "0" * 64).status_code == 200
    assert client.post(f"/sessions/{s2['id']}/finalize").status_code == 200
    assert batch(s2["id"], count=1).status_code == 409


def test_batch_concurrent_requests_never_oversubscribe():
    s = make_capped_session(20, max_active_leases=5)
    results: list[list[int]] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(8)

    def worker():
        try:
            barrier.wait()
            r = batch(s["id"], count=2)
            assert r.status_code == 200, r.text
            results.append([i["chunk_index"] for i in r.json()["leases"]])
        except Exception as exc:  # pragma: no cover - 仅用于暴露线程内失败
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    claimed = [idx for claimed_list in results for idx in claimed_list]
    # 无重复分配
    assert len(claimed) == len(set(claimed))
    # 有效租约总数恰好等于上限（5 个槽位，共 8*2 的需求）
    assert len(claimed) == 5
    st = client.get(f"/sessions/{s['id']}").json()
    assert st["active_leases"] == 5


def test_batch_unlimited_session_has_none_remaining():
    s = make_session(total_chunks=3)
    body = batch(s["id"], count=10).json()
    assert body["granted_count"] == 3  # 可领取分片耗尽，而非受槽位限制
    assert [i["chunk_index"] for i in body["leases"]] == [0, 1, 2]
    assert body["active"] == 3
    assert body["remaining_slots"] is None


# ---------------------------------------------------------------- 旧库迁移

def test_migration_adds_max_active_leases_to_legacy_db():
    import sqlite3

    legacy_path = os.path.join(tempfile.mkdtemp(), "legacy.db")
    conn = sqlite3.connect(legacy_path)
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, total_chunks INTEGER NOT NULL,
            file_digest TEXT NOT NULL, digest_algorithm TEXT NOT NULL DEFAULT 'sha256',
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL, finalized_at TEXT
        );
        CREATE TABLE chunks (
            session_id TEXT NOT NULL, chunk_index INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', lease_token TEXT,
            lease_expires_at TEXT, chunk_digest TEXT, completed_at TEXT,
            PRIMARY KEY (session_id, chunk_index)
        );
        CREATE TABLE chunk_dependencies (
            session_id TEXT NOT NULL, chunk_index INTEGER NOT NULL, depends_on INTEGER NOT NULL,
            PRIMARY KEY (session_id, chunk_index, depends_on)
        );
        INSERT INTO sessions (id, total_chunks, file_digest, created_at)
            VALUES ('legacy-session', 2, 'digest', '2026-01-01T00:00:00+00:00');
        INSERT INTO chunks (session_id, chunk_index) VALUES ('legacy-session', 0);
        """
    )
    conn.commit()
    conn.close()

    from app.db import _migrate

    conn = sqlite3.connect(legacy_path)
    conn.row_factory = sqlite3.Row
    _migrate(conn)
    _migrate(conn)  # 幂等：重复执行不报错
    columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
    assert "max_active_leases" in columns
    # 历史会话不限制并发，保持旧行为
    row = conn.execute(
        "SELECT max_active_leases FROM sessions WHERE id = 'legacy-session'"
    ).fetchone()
    assert row[0] is None
    conn.close()
