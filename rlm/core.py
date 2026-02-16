"""
core.py — Recursive Language Model with True Context Isolation.

Key design choice:
  `history` is a LOCAL variable inside each completion call, NOT
  an instance attribute.  This means every recursive call — whether
  from the parent or a child — runs with a completely fresh context.
  The parent never sees the child's intermediate errors; it only
  receives the final answer string.  This is what solves "Context Rot".
"""

import re
import sys
from typing import Optional
import litellm
from rich.console import Console
from rich.panel import Panel
from .sandbox import Sandbox

console = Console(file=sys.__stderr__)


class RLM:
    """
    Recursive Language Model.

    Loop:
        1. Send history to LLM  →  get a response (Thought + Code)
        2. If "Final Answer:" found  →  return it
        3. Extract ```python```/```repl``` block  →  execute in Sandbox
        4. Feed stdout back as "Observation"
        5. Repeat (up to max_iterations)

    Recursion:
        The sandbox is injected with a helper function `rlm_query(prompt)`
        that spawns a brand-new child context with bounded depth.
    """

    def __init__(
        self,
        model_name: str = "groq/llama-3.3-70b-versatile",
        max_iterations: int = 10,
        max_depth: int = 4,
        root_prompt: Optional[str] = None,
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.root_prompt = root_prompt
        self.verbose = verbose

    # ── public api ───────────────────────────────────────────────

    def completion(self, prompt: str, root_prompt: Optional[str] = None) -> str:
        """
        Solve *prompt* using the Thought → Code → Observation loop.

        IMPORTANT: `history` is local to each call.  A recursive child
        gets its own empty history, so the parent's context stays clean.
        """
        return self._completion(prompt=prompt, depth=1, root_prompt=root_prompt)

    def _completion(self, prompt: str, depth: int, root_prompt: Optional[str] = None) -> str:
        if depth > self.max_depth:
            return f"⚠️  Max recursion depth ({self.max_depth}) reached."

        indent = "  " * (depth - 1)
        depth_label = f"[depth {depth}]"

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
            {"role": "user", "content": self._direct_prompt(depth, root_prompt)},
        ]

        if self.verbose:
            # Explicitly log the proof of isolation
            console.print(
                f"{indent}[dim]Spawned Agent at Depth {depth}. "
                f"History size: {len(history)} (Fresh Context). "
                f"Task payload is in REPL variable `context`.[/dim]"
            )

        def child_query(child_prompt: str) -> str:
            return self._completion(prompt=child_prompt, depth=depth + 1)

        # ── fresh sandbox for THIS call ──────────────────────────
        # `rlm_query` lets the LLM call us recursively from code.
        # `context` carries the task payload outside LLM chat history.
        sandbox = Sandbox(extra_globals={"rlm_query": child_query, "context": prompt})

        for i in range(self.max_iterations):
            if self.verbose:
                console.print(f"\n{indent}[bold blue]{'─' * 40}[/bold blue]")
                console.print(
                    f"{indent}[bold blue]Iteration {i + 1}/{self.max_iterations} {depth_label}[/bold blue]"
                )

            # 1. LLM call
            response = litellm.completion(
                model=self.model_name,
                messages=history,
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""

            if self.verbose:
                console.print(
                    Panel(content, title=f"🤖 LLM Response {depth_label}", border_style="cyan")
                )

            history.append({"role": "assistant", "content": content})

            # 2. Check for final answer
            answer = self._extract_final_answer(content)
            if answer is not None:
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
                        Panel(code, title=f"⚡ Executing Code {depth_label}", border_style="yellow")
                    )

                output = sandbox.execute(code)

                if self.verbose:
                    console.print(
                        Panel(output, title=f"👁 Observation {depth_label}", border_style="dim")
                    )

                history.append({"role": "user", "content": f"Observation:\n{output}"})
            else:
                # No code and no final answer — nudge the model
                history.append(
                    {
                        "role": "user",
                        "content": (
                            "I did not see any code or a 'Final Answer:'. "
                            "Use ```python``` (or ```repl```) code when needed, and "
                            "read task details from the REPL variable `context`."
                        ),
                    }
                )

        return "⚠️  Max iterations reached without a Final Answer."

    # ── private helpers ──────────────────────────────────────────

    @staticmethod
    def _extract_code(text: str) -> Optional[str]:
        """Return the first ```python``` or ```repl``` block, or None."""
        match = re.search(r"```(?:python|repl)\s*(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def _extract_final_answer(text: str) -> Optional[str]:
        # Ignore code blocks so string literals like "Final Answer:" inside code
        # do not prematurely terminate the loop.
        text_without_code = re.sub(r"```(?:python|repl)?\s*.*?```", "", text, flags=re.DOTALL)
        match = re.search(r"(?im)^\s*Final Answer:\s*(.+)$", text_without_code)
        return match.group(1).strip() if match else None

    def _direct_prompt(self, depth: int, root_prompt: Optional[str]) -> str:
        if depth == 1:
            if root_prompt is not None:
                return root_prompt
            if self.root_prompt is not None:
                return self.root_prompt
        return (
            "Solve the task using Python execution.\n"
            "The full task payload is available only in the REPL variable `context`.\n"
            "Read `context`, compute what is needed, and end with `Final Answer:`."
        )

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a Recursive Language Model (RLM).\n"
            "You solve tasks by writing and executing Python code.\n"
            "You can recursively delegate with `rlm_query(subtask)`.\n\n"
            "Rules:\n"
            "1. Wrap code in ```python ... ``` or ```repl ... ``` blocks. The output will be\n"
            "   returned to you as an 'Observation'.\n"
            "2. You have a helper: `rlm_query(prompt)` — it spawns a NEW\n"
            "   RLM agent with a FRESH context to solve a sub-task and\n"
            "   returns the answer as a string.  Use it to decompose\n"
            "   complex problems!\n"
            "3. The task payload is in REPL variable `context` and may not appear in chat.\n"
            "4. Use `print()` to see results.\n"
            "5. When you have the final answer, output it on its own line\n"
            "   starting with 'Final Answer:'.\n"
        )
