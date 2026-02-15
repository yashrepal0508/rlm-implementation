"""
demo.py — Run a quick RLM demo.

Usage:
    uv run python demo.py

Configure the model via environment variable:
    RLM_MODEL=groq/llama-3.3-70b-versatile uv run python demo.py   # default
    RLM_MODEL=ollama/llama3 uv run python demo.py                  # local
    RLM_MODEL=gemini/gemini-2.0-flash uv run python demo.py        # needs GEMINI_API_KEY
"""

import os
from dotenv import load_dotenv
from rlm import RLM
from rich.console import Console

load_dotenv()

console = Console()


def main():
    model_name = os.getenv("RLM_MODEL", "groq/llama-3.3-70b-versatile")

    console.print(f"[bold]Starting RLM Demo with model:[/bold] {model_name}")
    console.print(
        "[dim]Using Groq by default. Set RLM_MODEL env var to change provider.[/dim]\n"
    )

    agent = RLM(model_name=model_name, verbose=True)

    # A multi-step task that benefits from code execution
    query = (
        "Calculate the 10th Fibonacci number. "
        "Then, use that number to calculate its square root."
    )

    console.print(f"[bold green]User Query:[/bold green] {query}\n")

    try:
        result = agent.completion(query)
        console.print(f"\n[bold purple]═══ Final Result ═══[/bold purple]")
        console.print(f"[bold]{result}[/bold]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        console.print(
            "[dim]Tip: If using Ollama, make sure it's running. "
            "If using an API, check your keys.[/dim]"
        )


if __name__ == "__main__":
    main()
