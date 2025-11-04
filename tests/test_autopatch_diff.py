from pathlib import Path
from meta.autopatch import render_diff, static_checks, _LAST_CONTEXT


def test_render_diff_basic():
    before = "a\nb\n"
    after = "a\nc\n"
    diff = render_diff(before, after, path="X")
    assert "-b" in diff and "+c" in diff


def test_static_checks_param_bounds(tmp_path):
    # 针对 scripts/spike_writer.py 的解码参数区段做越界值，期望 static_checks=False 并滚回
    rel = "scripts/spike_writer.py"
    tgt = Path(rel)
    original = tgt.read_text(encoding="utf-8")
    try:
        # 修改锚点内的 REPEAT_PENALTY 为 10.0（越界）
        start = "# AUTOPATCH DECODE PARAMS START"
        end = "# AUTOPATCH DECODE PARAMS END"
        s = original.find(start)
        e = original.find(end)
        assert s != -1 and e != -1 and e > s
        head = original[:s]
        body = original[s:e]
        tail = original[e:]
        body2 = body.replace("DECODE_REPEAT_PENALTY = 1.1", "DECODE_REPEAT_PENALTY = 10.0")
        modified = head + body2 + tail
        tgt.write_text(modified, encoding="utf-8")
        # 组装 _LAST_CONTEXT 以触发 import/API 比较（不使用）
        _LAST_CONTEXT.clear()
        _LAST_CONTEXT.update({"backups": {rel: str(tmp_path/"bak")}, "pre_sources": {rel: original}})
        assert static_checks([rel]) is False
    finally:
        tgt.write_text(original, encoding="utf-8")
