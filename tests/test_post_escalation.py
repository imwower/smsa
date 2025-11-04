from pathlib import Path


class _FakeSP:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)

    def run(self, *, topic=None, max_len=0, seed_text=""):
        # 每次调用返回队列头部（模拟 retune→relearn→patch 等阶段）
        if self._outcomes:
            return self._outcomes.pop(0)
        return {
            "text": "默认",
            "score": 0.9,
            "details": {"readability": 0.9, "context": 0.9},
            "attempts": 1,
            "best_action": "baseline",
            "actions": [],
        }


def test_escalation_retune_relearn(tmp_path, monkeypatch):
    # 第一次：tokens 少且 overall 低；第二次（relearn 后）达标
    fail = {
        "text": "短",
        "score": 0.42,
        "details": {"readability": 0.3, "context": 0.3},
        "attempts": 3,
        "best_action": "decode:temperature↓",
        "actions": ["decode:temperature↓"],
    }
    ok = {
        "text": "合格文本合格文本" * 10,
        "score": 0.72,
        "details": {"readability": 0.7, "context": 0.7},
        "attempts": 2,
        "best_action": "decode:top_k↑",
        "actions": ["decode:top_k↑"],
    }
    # 替换 SupervisedPost 为假对象序列：第一次失败，第二次成功
    from tools import supervisor as sup
    outcomes = [fail, ok]
    monkeypatch.setattr(sup, "SupervisedPost", lambda attempts=3, thresholds=None: _FakeSP(outcomes))

    # 运行 run_post
    from scripts.daemon import run_post
    res = run_post(120, topic_hint="测试", attempts=3, post_thresholds={"overall": 0.6, "self_explain": 0.4})
    assert res["post_retuned"] is True
    assert res["post_relearned"] is True
    # 达标后不再触发后续阶段
    assert res.get("post_patched") in (False, None)


def test_decode_patch_rollback(tmp_path, monkeypatch):
    # 三次都低，从而触发 patch；让 autopatch 返回失败
    fail = {
        "text": "短",
        "score": 0.40,
        "details": {"readability": 0.3, "context": 0.3},
        "attempts": 3,
        "best_action": "decode:temperature↓",
        "actions": ["decode:temperature↓"],
    }
    from tools import supervisor as sup
    monkeypatch.setattr(sup, "SupervisedPost", lambda attempts=3, thresholds=None: _FakeSP([fail, fail, fail, fail]))
    # autopatch 返回失败
    from meta import autopatch as ap
    monkeypatch.setattr(ap, "safe_apply_and_eval", lambda pid, kind="post": (False, None))

    from scripts.daemon import run_post
    res = run_post(100, topic_hint="测试", attempts=3, post_thresholds={"overall": 0.8, "self_explain": 0.4})
    # 触发 patch 尝试，但状态为失败/回滚
    assert res["post_retuned"] is True
    assert res["post_relearned"] is True
    # 即使失败，也应包含 patched 字段（False）及 grown/cooldown 字段
    assert res.get("post_patched") in (False, None)
    assert "post_grown" in res and "post_cooldown" in res
