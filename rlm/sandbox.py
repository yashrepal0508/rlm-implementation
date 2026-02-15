"""
Sandbox — a local Python REPL executor with isolated state.

Each sandbox instance maintains its own namespace so that
variables created in one execution persist for subsequent
executions within the *same* sandbox, but are invisible to
other sandboxes (e.g. a parent vs. child recursive call).
"""

import sys
import io
import traceback
from typing import Any, Dict


class Sandbox:
    """
    Executes Python code strings via `exec()` and captures stdout/stderr.
    A shared namespace (`self.namespace`) is kept between calls so that
    the agent can build on previous outputs within a single task.
    """

    def __init__(self, extra_globals: Dict[str, Any] | None = None):
        """
        Args:
            extra_globals: Additional names to inject into the execution
                           namespace (e.g. `rlm_query` for recursion).
        """
        self.namespace: Dict[str, Any] = {}
        if extra_globals:
            self.namespace.update(extra_globals)

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
            return output if output else "[No output]"
        except Exception:
            traceback.print_exc(file=captured)
            return captured.getvalue()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
