from pathlib import Path
from meta.autopatch import apply_patch, revert, static_checks, render_diff, _LAST_CONTEXT


def test_render_diff_basic():
    before = "a\nB\n"
    after = "a\nC\n"
    diff = render_diff(before, after, path="x.txt")
    assert "--- x.txt(before)" in diff and "+++ x.txt(after)" in diff
    assert "-B" in diff and "+C" in diff


def test_decode_topk_alias_and_static_checks_roundtrip():
    changed, backups = apply_patch("decode_topk_80")
    # touch an out-of-range value to trigger static check failure
    for rel in changed:
        full = Path(rel)
        if full.name == "spike_writer.py":
            text = full.read_text(encoding="utf-8")
            text = text.replace("DECODE_TOP_K = 80", "DECODE_TOP_K = 400")
            full.write_text(text, encoding="utf-8")
    ok = static_checks(changed)
    assert not ok  # should have reverted due to out-of-range top-k
    # Ensure we can revert explicitly too (no-op if already reverted)
    revert(backups)

