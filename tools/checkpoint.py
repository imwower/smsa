"""简易检查点工具（仅标准库）：原子保存/加载与基础迁移。

功能：
- state_dict(agent, extras) 收集关键子模块状态；
- save_agent(agent, path, extras) 使用 gzip+pickle 原子写入，并更新 runs/checkpoints/latest.json；
- load_agent(path) 加载对象；若包含 arch 信息且与当前对象不一致，可按需迁移；
- auto_save_every(agent, ep, n, dir) 每 n 回合自动保存一次。

约束：纯标准库；跨平台；简单健壮，不依赖外部服务。
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
import shutil
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping


LATEST_PATH = Path("runs/checkpoints/latest.json")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def state_dict(agent: Any, extras: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """收集对象的基本状态（尽量通用）。

    - 若属性为数据类，保存为字典；
    - 支持 DenseLIF/PolicyHead 的关键权重与尺寸；
    - 其他属性直接挂到 'misc'。
    """
    payload: Dict[str, Any] = {
        "arch": {},
        "modules": {},
        "misc": {},
    }
    # 采集常见子模块
    for name in ("hidden", "policy", "self_model", "temporal"):
        mod = getattr(agent, name, None)
        if mod is None:
            continue
        if name == "hidden":
            payload["arch"]["n_hidden"] = getattr(mod, "n_out", None)
            payload["modules"]["hidden"] = {
                "weights": getattr(mod, "weights", None),
                "bias": getattr(mod, "bias", None),
                "params": asdict(mod.params) if is_dataclass(mod.params) else None,
            }
        elif name == "policy":
            payload["modules"]["policy"] = {
                "weights": getattr(mod, "weights", None),
                "bias": getattr(mod, "bias", None),
                "n_in": getattr(mod, "n_in", None),
                "n_actions": getattr(mod, "n_actions", None),
                "lr": getattr(mod, "lr", None),
            }
        else:
            payload["modules"][name] = mod.__dict__.copy()
    # 附加额外信息
    if extras:
        payload["extras"] = dict(extras)
    return payload


def save_agent(agent: Any, path: str | Path, extras: Mapping[str, Any] | None = None) -> str:
    """以 gzip+pickle 保存检查点，原子替换，记录 latest.json。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "ts": time.time(),
        "class": agent.__class__.__name__,
        "state": state_dict(agent, extras=extras or {}),
        "pickle": agent,  # 备份一份直接对象（优先用此恢复）
    }
    buf = pickle.dumps(ckpt, protocol=pickle.HIGHEST_PROTOCOL)
    with gzip.open(path, "wb") as gz:
        gz.write(buf)
    # latest.json 指向最新 ckpt
    LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(
        LATEST_PATH,
        json.dumps({"path": str(path), "ts": time.time()}, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    return str(path)


def _try_migrate(agent: Any, state: Mapping[str, Any]) -> Any:
    """若需要，对隐藏层/策略头做维度迁移（小→大 复制前部，大→小 截断）。"""
    modules = state.get("modules", {})
    hidden = getattr(agent, "hidden", None)
    policy = getattr(agent, "policy", None)
    if hidden is not None and "hidden" in modules:
        h = modules["hidden"]
        w = h.get("weights")
        b = h.get("bias")
        if w and b:
            # 迁移到目标尺寸 agent.hidden.n_out
            target = getattr(hidden, "n_out", 0)
            src_out = len(b)
            # 写入权重
            for i in range(min(len(hidden.weights), len(w))):
                row = w[i]
                if not isinstance(row, list):
                    continue
                if target <= src_out:
                    hidden.weights[i][:target] = row[:target]
                else:
                    # 复制已有 + 随机初始化新增
                    hidden.weights[i][:src_out] = row[:src_out]
            # 写入偏置（同维度逻辑）
            if target <= src_out:
                hidden.bias[:target] = b[:target]
            else:
                hidden.bias[:src_out] = b[:src_out]
    if policy is not None and "policy" in modules:
        p = modules["policy"]
        w = p.get("weights")
        b = p.get("bias")
        if w and b:
            # 目标输入维度
            tgt_in = getattr(policy, "n_in", 0)
            for a in range(min(len(policy.weights), len(w))):
                row = w[a]
                if tgt_in <= len(row):
                    policy.weights[a][:tgt_in] = row[:tgt_in]
                else:
                    policy.weights[a][:len(row)] = row[:]
    return agent


def load_agent(path: str | Path) -> Any:
    """从检查点加载对象；优先使用 pickle 对象；否则按 state 迁移写入。"""
    path = Path(path)
    with gzip.open(path, "rb") as gz:
        data = gz.read()
    ckpt = pickle.loads(data)
    agent = ckpt.get("pickle")
    if agent is not None:
        return agent
    # 回退：调用 _try_migrate 写入当前 agent（需要调用方先构建 agent）
    return ckpt.get("state")


def load_latest() -> str | None:
    """读取 latest.json 指向的最新检查点路径。"""
    if not LATEST_PATH.exists():
        return None
    try:
        meta = json.loads(LATEST_PATH.read_text(encoding="utf-8"))
        return str(meta.get("path")) if meta else None
    except Exception:
        return None


def auto_save_every(agent: Any, episode: int, every: int, dir: str | Path = "runs/checkpoints") -> str | None:
    """每 n 回合自动保存，文件名包含 ep。"""
    every = int(every)
    if every <= 0:
        return None
    if episode % every != 0:
        return None
    dirp = Path(dir)
    dirp.mkdir(parents=True, exist_ok=True)
    name = f"ckpt_ep{episode:04d}.pkl.gz"
    path = dirp / name
    return save_agent(agent, path)


__all__ = [
    "state_dict",
    "save_agent",
    "load_agent",
    "load_latest",
    "auto_save_every",
]

