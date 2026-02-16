import unittest
from types import SimpleNamespace
from unittest.mock import patch
import sys
import types

if "litellm" not in sys.modules:
    sys.modules["litellm"] = types.SimpleNamespace(completion=None)

if "rich.console" not in sys.modules:
    rich_module = types.ModuleType("rich")
    console_module = types.ModuleType("rich.console")
    panel_module = types.ModuleType("rich.panel")

    class Console:
        def __init__(self, *args, **kwargs):
            pass

        def print(self, *args, **kwargs):
            pass

    class Panel:
        def __init__(self, renderable, *args, **kwargs):
            self.renderable = renderable

        def __str__(self):
            return str(self.renderable)

    console_module.Console = Console
    panel_module.Panel = Panel
    sys.modules["rich"] = rich_module
    sys.modules["rich.console"] = console_module
    sys.modules["rich.panel"] = panel_module

from rlm.core import RLM
from rlm.sandbox import Sandbox


def _response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


class TestRLMBehavior(unittest.TestCase):
    def test_prompt_payload_not_in_message_history(self):
        captured_messages = []

        def fake_completion(model, messages, temperature):
            captured_messages.append(messages)
            return _response("Final Answer: done")

        agent = RLM(verbose=False)
        prompt_payload = "TOP SECRET TASK PAYLOAD"
        root_prompt = "Solve using context."

        with patch("rlm.core.litellm.completion", side_effect=fake_completion):
            result = agent.completion(prompt_payload, root_prompt=root_prompt)

        self.assertEqual(result, "done")
        self.assertEqual(captured_messages[0][1]["content"], root_prompt)
        rendered_history = " ".join(
            message["content"] for message in captured_messages[0] if "content" in message
        )
        self.assertNotIn(prompt_payload, rendered_history)

    def test_max_depth_guard_in_recursive_query(self):
        captured_messages = []
        responses = [
            _response("```python\nprint(rlm_query('child task'))\n```"),
            _response("Final Answer: done"),
        ]

        def fake_completion(model, messages, temperature):
            captured_messages.append(messages)
            return responses[len(captured_messages) - 1]

        agent = RLM(verbose=False, max_iterations=2, max_depth=1)
        with patch("rlm.core.litellm.completion", side_effect=fake_completion):
            result = agent.completion("root task")

        self.assertEqual(result, "done")
        second_call_messages = captured_messages[1]
        joined = " ".join(
            message["content"] for message in second_call_messages if "content" in message
        )
        self.assertIn("Max recursion depth (1) reached", joined)

    def test_final_answer_inside_code_block_is_ignored(self):
        calls = {"count": 0}

        def fake_completion(model, messages, temperature):
            calls["count"] += 1
            if calls["count"] == 1:
                # Includes "Final Answer:" inside code only; should not terminate.
                return _response(
                    "```python\n"
                    "result = 5\n"
                    "def get_final_answer(x):\n"
                    "    return f\"Final Answer: {x}\"\n"
                    "print(get_final_answer(result))\n"
                    "```"
                )
            return _response("Final Answer: 5")

        agent = RLM(verbose=False, max_iterations=3)
        with patch("rlm.core.litellm.completion", side_effect=fake_completion):
            result = agent.completion("compute 5")

        self.assertEqual(result, "5")
        self.assertEqual(calls["count"], 2)


class TestSandboxSafety(unittest.TestCase):
    def test_allows_safe_math_import(self):
        sandbox = Sandbox()
        output = sandbox.execute("import math\nprint(math.sqrt(16))")
        self.assertIn("4.0", output)

    def test_blocks_disallowed_import(self):
        sandbox = Sandbox()
        output = sandbox.execute("import os")
        self.assertIn("ImportError", output)
        self.assertIn("blocked", output)


if __name__ == "__main__":
    unittest.main()
