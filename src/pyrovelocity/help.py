import contextlib
import io
from types import ModuleType
from typing import Callable


__all__ = ["internal_help"]


def internal_help(obj: Callable | ModuleType):
    """Generate help text for a callable or module.

    Args:
        obj (Callable | ModuleType): The object to generate help text for.
    """
    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        help(obj)

    help_text = captured_output.getvalue()
    lines = help_text.split("\n")

    processed_lines = []
    for line in lines:
        if line.startswith("FILE"):
            break
        processed_lines.append(line)

    print("\n".join(processed_lines))
