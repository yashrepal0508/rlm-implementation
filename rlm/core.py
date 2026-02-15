"""
core.py — Recursive Language Model with True Context Isolation.

Key design choice:
  `message_history` is a LOCAL variable inside `completion()`, NOT
  an instance attribute.  This means every recursive call — whether
  from the parent or a child — runs with a completely fresh context.
  The parent never sees the child's intermediate errors; it only
  receives the final answer string.  This is what solves "Context Rot".
"""

import re
import litellm
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from .sandbox import Sandbox

console = Console()


class RLM:
    """
    Recursive Language Model.

    Loop:
        1. Send history to LLM  →  get a response (Thought + Code)
        2. If "Final Answer:" found  →  return it
        3. Extract ```python``` block  →  execute in Sandbox
        4. Feed stdout back as "Observation"
        5. Repeat (up to max_iterations)

    Recursion:
        The sandbox is injected with a helper function `rlm_query(prompt)`
        that calls `self.completion(prompt)` — spawning a brand-new context.
    """

    def __init__(
        self,
        model_name: str = "groq/llama-3.3-70b-versatile",
        max_iterations: int = 10,
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.verbose = verbose
        # Track recursion depth for logging clarity
        self._depth = 0

    # ── public api ───────────────────────────────────────────────

    def completion(self, prompt: str) -> str:
        """
        Solve *prompt* using the Thought → Code → Observation loop.

        IMPORTANT: `history` is local to this call.  A recursive child
        gets its own empty history, so the parent's context stays clean.
        """
        self._depth += 1
        depth = self._depth
        indent = "  " * (depth - 1)
        depth_label = f"[depth {depth}]" if depth > 1 else ""

        if self.verbose:
            console.print(
                Panel(
                    f"{prompt}",
                    title=f"🚀 RLM Start {depth_label}",
                    border_style="green",
                )
            )

        # ── fresh context for THIS call ──────────────────────────
        history = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": prompt},
        ]

        # ── fresh sandbox for THIS call ──────────────────────────
        # `rlm_query` lets the LLM call us recursively from code.
        sandbox = Sandbox(extra_globals={"rlm_query": self.completion})

        try:
            for i in range(self.max_iterations):
                if self.verbose:
                    console.print(
                        f"\n{indent}[bold blue]{'─'*40}[/bold blue]"
                    )
                    console.print(
                        f"{indent}[bold blue]Iteration {i+1}/{self.max_iterations} {depth_label}[/bold blue]"
                    )

                # 1. LLM call
                response = litellm.completion(
                    model=self.model_name,
                    messages=history,
                    temperature=0.0,
                )
                content: str = response.choices[0].message.content

                if self.verbose:
                    console.print(
                        Panel(content, title="🤖 LLM Response", border_style="cyan")
                    )

                history.append({"role": "assistant", "content": content})

                # 2. Check for final answer
                if "Final Answer:" in content:
                    answer = content.split("Final Answer:")[-1].strip()
                    if self.verbose:
                        console.print(
                            Panel(
                                answer,
                                title=f"✅ Final Answer {depth_label}",
                                border_style="green",
                            )
                        )
                    return answer

                # 3. Extract & execute code
                code = self._extract_code(content)
                if code:
                    if self.verbose:
                        console.print(
                            Panel(code, title="⚡ Executing Code", border_style="yellow")
                        )

                    output = sandbox.execute(code)

                    if self.verbose:
                        console.print(
                            Panel(output, title="👁 Observation", border_style="dim")
                        )

                    history.append(
                        {"role": "user", "content": f"Observation:\n{output}"}
                    )
                else:
                    # No code and no final answer — nudge the model
                    history.append(
                        {
                            "role": "user",
                            "content": (
                                "I did not see any code or a 'Final Answer:'. "
                                "Please write Python code in a ```python``` block, "
                                "or provide your Final Answer."
                            ),
                        }
                    )

            return "⚠️  Max iterations reached without a Final Answer."

        finally:
            self._depth -= 1

    # ── private helpers ──────────────────────────────────────────

    @staticmethod
    def _extract_code(text: str) -> Optional[str]:
        """Return the first ```python ... ``` block, or None."""
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a Recursive Language Model (RLM).\n"
            "You solve tasks by writing and executing Python code.\n\n"
            "Rules:\n"
            "1. Wrap code in ```python ... ``` blocks. The output will be\n"
            "   returned to you as an 'Observation'.\n"
            "2. You have a helper: `rlm_query(prompt)` — it spawns a NEW\n"
            "   RLM agent with a FRESH context to solve a sub-task and\n"
            "   returns the answer as a string.  Use it to decompose\n"
            "   complex problems!\n"
            "3. Use `print()` to see results.\n"
            "4. When you have the final answer, output it on its own line\n"
            "   starting with 'Final Answer:'.\n"
        )
