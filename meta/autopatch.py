"""安全补丁引擎（仅标准库）。

功能概述
- 仅允许在白名单锚点内修改三处代码：
  - snn/lif.py：# AUTOPATCH SURROGATE START/END（替代导数/表达式片段）
  - snn/dense.py：# AUTOPATCH DEFAULTS START/END（eta_e/lam_e 默认值）
  - meta/autoadapt.py：# AUTOPATCH CANDIDATES START/END（候选动作表）
- 两种补丁策略：
  1) AST 替换（将锚点区域视为一个可解析的函数体或常量表达式并替换）
  2) 基于锚点的安全字符串替换（完全不触碰锚点外的任何字符）
- 提供 apply_patch()/revert()/static_checks()/smoke_test()/ab_evaluate()。
  任何一步失败都会自动 revert，并把细节写入 runs/autopatch.log。

注意
- 引擎不依赖外部服务；所有操作使用标准库完成。
"""

from __future__ import annotations

import ast
import contextlib
import datetime as _dt
import io
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# 常量与锚点定义
SURROGATE_FILE = Path("snn/lif.py")
DEFAULTS_FILE = Path("snn/dense.py")
CANDIDATES_FILE = Path("meta/autoadapt.py")
ANCHORS = {
    str(SURROGATE_FILE): ("# AUTOPATCH SURROGATE START", "# AUTOPATCH SURROGATE END"),
    str(DEFAULTS_FILE): ("# AUTOPATCH DEFAULTS START", "# AUTOPATCH DEFAULTS END"),
    str(CANDIDATES_FILE): ("# AUTOPATCH CANDIDATES START", "# AUTOPATCH CANDIDATES END"),
}

LOG_PATH = Path("runs/autopatch.log")
STATE_PATH = Path("runs/autopatch_state.json")
BACKUP_ROOT = Path("runs/autopatch_backups")


def _log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{timestamp}] {message}\n")


@dataclass
class PatchRecipe:
    file: Path
    strategy: str  # "anchor-replace" | "ast-rewrite"
    content: str   # 替换后的锚点内容文本（或函数体/表达式）


# 示例内置补丁（可扩展）：
_RECIPES: Dict[str, List[PatchRecipe]] = {
    # 在 lif.py 的锚点中放入一个温和的 Sigmoid 替代导数表达式示例
    "surrogate:gentle": [
        PatchRecipe(
            file=SURROGATE_FILE,
            strategy="anchor-replace",
            content=(
                "# 使用平滑 Sigmoid 的替代导数表达式\n"
                "def fast_sigmoid_surrogate(u: float, slope: float = 2.0) -> float:\n"
                "    denom = 1.0 + slope * abs(u)\n"
                "    return slope / (denom * denom)\n"
            ),
        )
    ],
    # 切换为矩形窗替代导数：在 |u| <= (1/slope) 内为常数，否则为 0
    "surrogate:rect": [
        PatchRecipe(
            file=SURROGATE_FILE,
            strategy="anchor-replace",
            content=(
                "# 使用矩形窗的替代导数；窗口宽度 ~ 1/slope\n"
                "def fast_sigmoid_surrogate(u: float, slope: float = 2.0) -> float:\n"
                "    \"\"\"矩形窗替代导数：在小邻域给常数梯度。\"\"\"\n"
                "    width = 1.0 / max(1e-6, slope)\n"
                "    return (slope if -width <= u <= width else 0.0)\n"
            ),
        )
    ],
    # 在 dense.py 的锚点中设定 e-prop 默认学习率/衰减（需通过静态检查范围）
    "defaults:eta0.03_lam0.9": [
        PatchRecipe(
            file=DEFAULTS_FILE,
            strategy="anchor-replace",
            content=(
                "# e-prop 默认参数\n"
                "ETA_E_DEFAULT = 0.03\nLAM_E_DEFAULT = 0.90\n"
            ),
        )
    ],
    # 在 autoadapt.py 的锚点中调整候选动作集合
    "candidates:minimal": [
        PatchRecipe(
            file=CANDIDATES_FILE,
            strategy="anchor-replace",
            content=(
                "# 精简候选动作表\n"
                "DEFAULT_ACTIONS = [\n"
                "    'eta_up', 'eta_down', 'vth_up', 'vth_down'\n"
                "]\n"
            ),
        )
    ],
}


def _read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as fh:
        return fh.read()


def _write_text(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(data)


def _find_anchor_span(text: str, start_tag: str, end_tag: str) -> Tuple[int, int]:
    start = text.find(start_tag)
    end = text.find(end_tag)
    if start == -1 or end == -1 or end < start:
        return -1, -1
    # 把锚点内容定义为两行标记之间的全部字符
    start_idx = start + len(start_tag)
    end_idx = end
    return start_idx, end_idx


def _anchor_replace(full: str, start_tag: str, end_tag: str, new_block: str) -> str:
    s, e = _find_anchor_span(full, start_tag, end_tag)
    if s == -1:
        raise ValueError("未找到锚点，取消变更。")
    return full[:s] + "\n" + new_block.rstrip("\n") + "\n" + full[e:]


def _backup_path(ts: str, file: Path) -> Path:
    return BACKUP_ROOT / ts / (str(file) + ".bak")


def _snapshot_api_signatures(source: str) -> Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    """抓取文件中的顶层函数签名：name -> (args, kwonly)."""
    out: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return out
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            args = tuple(arg.arg for arg in node.args.args)
            kwonly = tuple(arg.arg for arg in node.args.kwonlyargs)
            out[node.name] = (args, kwonly)
        if isinstance(node, ast.AsyncFunctionDef):
            args = tuple(arg.arg for arg in node.args.args)
            kwonly = tuple(arg.arg for arg in node.args.kwonlyargs)
            out[node.name] = (args, kwonly)
    return out


def _collect_imports(source: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """返回 (import 语句列表, from-import 语句列表)。"""
    imports: List[str] = []
    froms: List[str] = []
    for line in source.splitlines():
        s = line.strip()
        if s.startswith("import "):
            imports.append(s)
        elif s.startswith("from ") and " import " in s:
            froms.append(s)
    return tuple(imports), tuple(froms)


_LAST_CONTEXT: Dict[str, object] = {}


def apply_patch(patch_id: str) -> Tuple[List[str], Dict[str, str]]:
    """应用指定补丁并返回 (changed_files, backups)。

    若锚点缺失或补丁策略非法，记录日志并抛出异常。
    """
    recipes = _RECIPES.get(patch_id)
    if not recipes:
        _log(f"apply_patch: 未知 patch_id={patch_id}")
        raise KeyError(f"不支持的 patch: {patch_id}")

    timestamp = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    changed: List[str] = []
    backups: Dict[str, str] = {}
    pre_sources: Dict[str, str] = {}

    for rec in recipes:
        path = rec.file
        full_path = ROOT / path
        if not full_path.exists():
            _log(f"apply_patch: 目标文件缺失: {path}")
            raise FileNotFoundError(str(path))

        before = _read_text(full_path)
        pre_sources[str(path)] = before
        start_tag, end_tag = ANCHORS.get(str(path), (None, None))
        if not start_tag or not end_tag:
            _log(f"apply_patch: 未配置锚点: {path}")
            raise ValueError(f"未配置锚点: {path}")

        # 备份
        bak_path = _backup_path(timestamp, path)
        bak_path.parent.mkdir(parents=True, exist_ok=True)
        _write_text(bak_path, before)
        backups[str(path)] = str(bak_path)

        # 仅在锚点内替换
        if rec.strategy == "anchor-replace":
            after = _anchor_replace(before, start_tag, end_tag, rec.content)
        elif rec.strategy == "ast-rewrite":
            # 将锚点内部视为一个独立源码片段，尝试替换为 rec.content
            s, e = _find_anchor_span(before, start_tag, end_tag)
            if s == -1:
                raise ValueError("未找到锚点，取消变更。")
            _ = ast.parse(rec.content)  # 语法校验
            after = before[:s] + "\n" + rec.content.rstrip("\n") + "\n" + before[e:]
        else:
            _log(f"apply_patch: 非法策略 {rec.strategy}")
            raise ValueError(f"非法策略: {rec.strategy}")

        _write_text(full_path, after)
        changed.append(str(path))

    # 缓存上下文，供后续 static/smoke/ab 使用
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LAST_CONTEXT.clear()
    _LAST_CONTEXT.update({
        "patch_id": patch_id,
        "timestamp": timestamp,
        "changed": changed,
        "backups": backups,
        "pre_sources": pre_sources,
    })
    with STATE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(_LAST_CONTEXT, fh, ensure_ascii=False, indent=2)

    _log(f"apply_patch: 已应用 patch {patch_id} → {changed}")
    return changed, backups


def revert(backups: Mapping[str, str]) -> None:
    """用备份回滚改动。backups: 相对路径 -> 备份文件绝对路径。"""
    for rel, bak in backups.items():
        src = Path(bak)
        dst = ROOT / rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    _log(f"revert: 已回滚 {list(backups)}")


def static_checks(changed_files: Sequence[str]) -> bool:
    """静态约束：
    - 禁止新增 import（比较前后）
    - 禁止改变公共 API 签名（比较顶层函数形参）
    - eta_e ∈ [1e-4, 0.1]，lam_e ∈ [0.5, 0.999]（若锚点中出现）
    失败将自动回滚。
    """
    ctx = dict(_LAST_CONTEXT)
    backups: Dict[str, str] = ctx.get("backups", {})  # type: ignore[assignment]
    pre_sources: Dict[str, str] = ctx.get("pre_sources", {})  # type: ignore[assignment]
    ok = True
    reasons: List[str] = []

    for rel in changed_files:
        full = ROOT / rel
        before = pre_sources.get(rel, "")
        try:
            after = _read_text(full)
        except OSError:
            ok = False
            reasons.append(f"无法读取修改后的文件: {rel}")
            continue

        # import 检查
        b_imp, b_from = _collect_imports(before)
        a_imp, a_from = _collect_imports(after)
        if len(a_imp) > len(b_imp) or len(a_from) > len(b_from):
            ok = False
            reasons.append(f"{rel} 引入新的 import 语句，禁止。")

        # API 签名检查
        b_api = _snapshot_api_signatures(before)
        a_api = _snapshot_api_signatures(after)
        if b_api != a_api:
            ok = False
            reasons.append(f"{rel} 顶层函数签名发生变化，禁止。")

        # 数值范围：在锚点中提取浮点数进行约束
        start_tag, end_tag = ANCHORS.get(rel, (None, None))
        if start_tag and end_tag:
            s, e = _find_anchor_span(after, start_tag, end_tag)
            if s != -1:
                block = after[s:e]
                # 粗略抓取浮点常量
                import re as _re
                floats = [float(x) for x in _re.findall(r"(?<![A-Za-z0-9_])([0-9]*\.?[0-9]+)", block)]
                # 若命名出现则更精确地校验：
                if "ETA_E_DEFAULT" in block:
                    vals = [float(x) for x in _re.findall(r"ETA_E_DEFAULT\s*=\s*([0-9]*\.?[0-9]+)", block)]
                    for v in vals:
                        if not (1e-4 <= v <= 0.1):
                            ok = False
                            reasons.append(f"eta_e 超界: {v}")
                if "LAM_E_DEFAULT" in block:
                    vals = [float(x) for x in _re.findall(r"LAM_E_DEFAULT\s*=\s*([0-9]*\.?[0-9]+)", block)]
                    for v in vals:
                        if not (0.5 <= v <= 0.999):
                            ok = False
                            reasons.append(f"lam_e 超界: {v}")

    if not ok:
        _log("static_checks: 失败 → 执行回滚。原因：" + "; ".join(reasons))
        with contextlib.suppress(Exception):
            revert(backups)
        return False

    _log("static_checks: 通过")
    return True


def smoke_test() -> bool:
    """快速子集测试：近似 pytest -k 'lif or eprop or policy'。

    实现：使用 unittest 默认加载器，按模块/类名筛选 lif/eprop/policy 相关用例。
    失败将自动回滚。
    """
    ctx = dict(_LAST_CONTEXT)
    backups: Dict[str, str] = ctx.get("backups", {})  # type: ignore[assignment]

    import unittest

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 仅挑选这几组模块
    targets = [
        "tests.test_lif",
        "tests.test_eprop",
        "tests.test_policy",
    ]

    for mod in targets:
        try:
            suite.addTests(loader.loadTestsFromName(mod))
        except Exception:
            # 某些仓库可能缺失部分测试；忽略加载错误
            continue

    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=1, failfast=True)
    result = runner.run(suite)
    ok = result.wasSuccessful()
    if not ok:
        _log("smoke_test: 失败 → 回滚。输出：" + stream.getvalue())
        with contextlib.suppress(Exception):
            revert(backups)
        return False

    _log("smoke_test: 通过")
    return True


def _measure_small_lm(seed: int = 0) -> Tuple[float, float]:
    """返回 (score, energy)：其中 score 越大越好，约为 -Δppl。"""
    try:
        from scripts.snn_text_lm import train_lines  # 延迟导入
    except Exception as exc:  # pragma: no cover - 非关键路径
        _log(f"ab_evaluate: 无法导入 train_lines: {exc}")
        return 0.0, 0.0
    stats = train_lines(num_lines=120, seed=seed, sampler=None, valid_interval=60)
    delta_ppl = float(stats.get("delta_ppl", 0.0) or 0.0)
    avg_spikes = float(stats.get("avg_spikes", 0.0) or 0.0)
    score = -delta_ppl
    return score, avg_spikes


def ab_evaluate() -> Tuple[float, float]:
    """执行 A/B 评估，返回 (delta_score, delta_energy)。

    实现：
    - 使用 LM 的 train_lines 作为快速代理指标；先在 baseline（回滚）下测量，
      再在已打补丁版本下测量；两者使用相同 seed 以对齐随机性。
    - 如果补丁版分数低于 baseline（delta_score < 0），自动回滚。
    """
    ctx = dict(_LAST_CONTEXT)
    backups: Dict[str, str] = ctx.get("backups", {})  # type: ignore[assignment]
    changed: List[str] = ctx.get("changed", [])  # type: ignore[assignment]
    if not backups:
        _log("ab_evaluate: 缺少备份上下文，跳过。")
        return 0.0, 0.0

    seed = int(_dt.datetime.utcnow().timestamp()) & 0xFFFF

    # 先回滚测 baseline
    with contextlib.suppress(Exception):
        revert(backups)
    base_score, base_energy = _measure_small_lm(seed=seed)

    # 重新应用补丁（从磁盘状态重建）
    patch_id = ctx.get("patch_id")
    try:
        if isinstance(patch_id, str):
            apply_patch(patch_id)
    except Exception as exc:
        _log(f"ab_evaluate: 重新应用补丁失败: {exc}")
        return 0.0, 0.0

    patch_score, patch_energy = _measure_small_lm(seed=seed)
    delta_score = patch_score - base_score
    delta_energy = patch_energy - base_energy

    if delta_score < 0.0:
        _log(
            f"ab_evaluate: 退化 (Δscore={delta_score:.4f}, Δenergy={delta_energy:.2f}) → 回滚"
        )
        with contextlib.suppress(Exception):
            revert(_LAST_CONTEXT.get("backups", {}))  # type: ignore[arg-type]
    else:
        _log(
            f"ab_evaluate: 提升 (Δscore={delta_score:.4f}, Δenergy={delta_energy:.2f})"
        )

    return delta_score, delta_energy


# 便捷流水线（可选）：一键执行所有步骤。
def safe_apply_and_eval(patch_id: str) -> Tuple[bool, Tuple[float, float] | None]:
    """封装：apply → static → smoke → A/B。任一步失败自动回滚并返回 False。"""
    try:
        changed, _ = apply_patch(patch_id)
    except Exception as exc:
        _log(f"pipeline: apply 失败: {exc}")
        return False, None
    if not static_checks(changed):
        return False, None
    if not smoke_test():
        return False, None
    delta = ab_evaluate()
    return (delta[0] >= 0.0), delta


__all__ = [
    "apply_patch",
    "revert",
    "static_checks",
    "smoke_test",
    "ab_evaluate",
    "safe_apply_and_eval",
]
