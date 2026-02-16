"""
Sandbox — a local Python REPL executor with isolated state.

Each sandbox instance maintains its own namespace so that
variables created in one execution persist for subsequent
executions within the *same* sandbox, but are invisible to
other sandboxes (e.g. a parent vs. child recursive call).
"""

import builtins
import io
import sys
import traceback
from typing import Any, Dict, Optional


class Sandbox:
    """
    Executes Python code strings via `exec()` and captures stdout/stderr.
    A shared namespace (`self.namespace`) is kept between calls so that
    the agent can build on previous outputs within a single task.
    """

    _ALLOWED_IMPORT_ROOTS = {
        "collections",
        "decimal",
        "fractions",
        "functools",
        "itertools",
        "math",
        "random",
        "re",
        "statistics",
    }

    _SAFE_BUILTINS = {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "Exception",
        "filter",
        "float",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "ValueError",
        "zip",
    }

    def __init__(
        self,
        extra_globals: Optional[Dict[str, Any]] = None,
        max_output_chars: int = 8_000,
    ):
        """
        Args:
            extra_globals: Additional names to inject into the execution
                           namespace (e.g. `rlm_query` for recursion).
        """
        self.max_output_chars = max_output_chars
        self.namespace: Dict[str, Any] = {"__builtins__": self._build_builtins()}
        if extra_globals:
            self.namespace.update(extra_globals)

    def _safe_import(self, name: str, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root not in self._ALLOWED_IMPORT_ROOTS:
            raise ImportError(
                f"Import '{name}' is blocked in Sandbox. "
                f"Allowed roots: {sorted(self._ALLOWED_IMPORT_ROOTS)}"
            )
        return builtins.__import__(name, globals, locals, fromlist, level)

    def _build_builtins(self) -> Dict[str, Any]:
        allowed = {name: getattr(builtins, name) for name in self._SAFE_BUILTINS}
        allowed["__import__"] = self._safe_import
        return allowed

    def _truncate_output(self, output: str) -> str:
        if len(output) <= self.max_output_chars:
            return output
        suffix = "\n...[output truncated]"
        return output[: self.max_output_chars] + suffix

    def execute(self, code: str) -> str:
        """
        Execute *code* and return everything written to stdout/stderr.
        Errors are captured as tracebacks rather than raised.
        """
        old_stdout, old_stderr = sys.stdout, sys.stderr
        captured = io.StringIO()
        sys.stdout = captured
        sys.stderr = captured

        try:
            exec(code, self.namespace)
            output = captured.getvalue()
            if not output:
                return "[No output]"
            return self._truncate_output(output)
        except Exception:
            traceback.print_exc(file=captured)
            return self._truncate_output(captured.getvalue())
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
