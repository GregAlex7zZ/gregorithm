"""Guard the scientific tests against vacuous floating-point comparisons."""

from __future__ import annotations

import re
from pathlib import Path


def test_every_approx_states_an_explicit_tolerance() -> None:
    """pytest.approx defaults to abs=1e-12, which accepts any D of order 1e-14 m²/s.

    Passing only ``abs`` disables the relative tolerance, so ``abs=0`` needs an explicit ``rel``.
    """
    offenders = []
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"pytest\.approx\(", text):
            depth, index = 1, match.end()
            while depth:
                depth += (text[index] == "(") - (text[index] == ")")
                index += 1
            call = text[match.end() : index]
            if "abs=" not in call or ("abs=0" in call.replace(" ", "") and "rel=" not in call):
                offenders.append(f"{path.name}:{text.count(chr(10), 0, match.start()) + 1}")
    assert not offenders, offenders
