"""Shared label rendering for the manifold figures.

Two bugs this fixes:

1. CJK rendered as tofu. matplotlib 3.5 has no per-glyph font fallback, so a
   label mixing Latin and Chinese cannot be drawn by one monospace font that
   lacks CJK. Noto Sans CJK covers BOTH, so any label containing CJK is drawn
   entirely in Noto and the rest stay monospace.

2. Invisible tokens printed as escape codes. `repr()` turns a zero-width space
   into the seven characters \\u200b, which is not what the model emitted and
   reads as a bug in the figure. Invisible and control characters are now named
   explicitly with ASCII markers -- ASCII on purpose, so the fix cannot itself
   introduce a missing glyph.
"""
import unicodedata
from matplotlib import font_manager as fm

CJK_FONT = "Noto Sans CJK JP"
_NAMED = {" ": "[sp]", "​": "[zwsp]", "‌": "[zwnj]", "‍": "[zwj]",
          "﻿": "[bom]", "\n": "[nl]", "\t": "[tab]", "\r": "[cr]",
          "\xa0": "[nbsp]"}


def register(path="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    try:
        fm.fontManager.addfont(path)
        return True
    except Exception:
        return False


def has_cjk(s):
    return any("⺀" <= ch <= "鿿" or "가" <= ch <= "힯"
               or "＀" <= ch <= "￯" for ch in s)


def pretty(s):
    """A token as a reader can actually see it, quoted."""
    out = []
    for ch in s:
        if ch in _NAMED:
            out.append(_NAMED[ch])
        elif unicodedata.category(ch) in ("Cc", "Cf", "Zs", "Zl", "Zp"):
            out.append("[u+%04x]" % ord(ch))
        else:
            out.append(ch)
    return "'" + "".join(out) + "'"


def family(s):
    """Whole-label font choice; Noto covers Latin too, so mixing is safe."""
    return CJK_FONT if has_cjk(s) else "monospace"
