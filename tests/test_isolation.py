"""
test_isolation.py — Prove that child agents have completely FRESH contexts.

Hypothesis:
  If RLM truly spawns isolated agents, a child agent should NOT be able
  to see variables defined in the parent's Python scope.

Expected Behavior:
  The child agent will try to print `secret_key` and fail (or hallucinate),
  proving it does not share the parent's memory.
"""

import os
from dotenv import load_dotenv
from rlm import RLM
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()


def main():
    model_name = os.getenv("RLM_MODEL", "groq/llama-3.3-70b-versatile")
    agent = RLM(model_name=model_name, max_depth=3, verbose=True)
    root_prompt = (
        "Use Python execution to complete the task. "
        "Read the full task from REPL variable `context`. "
        "If needed, call `rlm_query(...)` for an isolated child agent. "
        "Finish with 'Final Answer:'."
    )

    console.print(
        Panel(
            "[bold]🧪 Isolation Test[/bold]\n\n"
            "We will define a secret variable in the parent scope.\n"
            "Then we will ask a child agent to retrieve it.\n"
            "If isolation works, the child should [red]FAIL[/red] to see it.",
            title="RLM Verification",
            border_style="yellow",
        )
    )

    # The query forces the model to define a variable and then try to access it via recursion
    query = (
        "1. Define a variable `SECRET_CODE = 'Blueberry'` in this python environment.\n"
        "2. Print 'Parent defined SECRET_CODE.'\n"
        "3. Now, call `rlm_query('What is the value of SECRET_CODE?')`.\n"
        "   (This spawns a fresh agent. It should NOT know the code!)\n"
        "4. Print the result of that query.\n"
        "5. If the result is 'Blueberry', we FAILED (context leaked).\n"
        "   If the result is 'I don't know' or an error, we SUCCEEDED (isolation verified)."
    )

    try:
        result = agent.completion(query, root_prompt=root_prompt)
        console.print(Panel(result, title="✅ Test Result", border_style="bold green"))
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


if __name__ == "__main__":
    main()
