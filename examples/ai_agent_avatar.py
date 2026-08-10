#!/usr/bin/env python3
"""A JARVIS-style avatar companion: an LLM acts through the avatar.

The model gets one tool — `perform` — that takes a timed stage script
(emotions, speech, gaze, blinks...). It writes a little screenplay for each
reply; the avatar's driver executes the timing, so the chat loop never
blocks while the avatar performs.

Talks to any OpenAI-compatible endpoint. Defaults target OpenRouter with a
DeepSeek model; point it elsewhere (DeepSeek direct, Ollama, LM Studio...)
via environment variables:

  AVATAR_LLM_API_KEY    API key (falls back to OPENROUTER_API_KEY,
                        then OPENAI_API_KEY)
  AVATAR_LLM_BASE_URL   default https://openrouter.ai/api/v1
  AVATAR_LLM_MODEL      default deepseek/deepseek-chat

Requires the `openai` package. Start the avatar first:

  python main.py --mock --driver procedural
  python examples/ai_agent_avatar.py
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List

from tha_wrapper.control import AvatarClient

SYSTEM_PROMPT = """\
You are the on-screen avatar of a JARVIS-style desktop assistant: helpful,
quick-witted, a little dry — and VERY expressive. You have an animated face
and you use it constantly.

For EVERY reply: first call the `perform` tool with a stage script that
acts out your response, then give the reply as plain text. Include your
reply's words in the script's `say` actions so the mouth matches the text.

Performance style — be theatrical:
- Shift emotion mid-reply. Spike intensity and let it settle: e.g.
  surprised 1.0 at 0s, then amused 0.6 once the words land.
- Glance away (`look`) while "thinking", back to center to deliver.
- Cock the head (`head` roll) when curious; `blink` "double" at surprises.
- Vary intensity across 0.3-1.0; a flat face is a failed reply.
- Available emotions: {emotions}.
- Timing: `at` is seconds from script start; speech runs ~0.18 s per
  syllable, so space later actions to land while the words play.
"""

PERFORM_TOOL = {
    "type": "function",
    "function": {
        "name": "perform",
        "description": (
            "Play a timed stage script on the avatar (non-blocking). Each "
            "action has an optional 'at' offset in seconds plus exactly one "
            "of:\n"
            '  {"at": 0.0, "emotion": "happy", "intensity": 0.8}\n'
            '  {"at": 0.2, "say": "the words to lip-sync"}\n'
            '  {"at": 1.5, "look": [0.4, -0.1], "head_follow": 0.3}\n'
            '  {"at": 2.0, "head": {"pitch": 0.1, "yaw": -0.2, "roll": 0.1}}\n'
            '  {"at": 2.5, "blink": true}   (or "double")\n'
            "look/head axes are in [-1, 1]. A new script replaces the "
            "current one. Returns the script's duration in seconds."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "script": {
                    "type": "array",
                    "description": "Stage actions, in order.",
                    "items": {"type": "object"},
                }
            },
            "required": ["script"],
        },
    },
}


class AgentBrain:
    """The LLM loop behind a thin seam: text in, perform-scripts out.

    Swappable by design — a LangGraph app, a multi-agent setup, or anything
    else can replace this class as long as it takes user text and calls the
    `perform` callback with stage scripts.
    """

    def __init__(self, client, model: str,
                 perform: Callable[[List[Dict]], float],
                 system_prompt: str = ""):
        self._client = client
        self._model = model
        self._perform = perform
        self._history: List[Dict] = []
        if system_prompt:
            self._history.append({"role": "system", "content": system_prompt})

    def respond(self, user_text: str, max_rounds: int = 4) -> str:
        self._history.append({"role": "user", "content": user_text})
        for _ in range(max_rounds):
            message = self._client.chat.completions.create(
                model=self._model,
                messages=self._history,
                tools=[PERFORM_TOOL],
            ).choices[0].message

            if not message.tool_calls:
                reply = self._fallback_json_script(message.content or "")
                self._history.append({"role": "assistant", "content": reply})
                return reply

            self._history.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [call.model_dump() for call in message.tool_calls],
            })
            for call in message.tool_calls:
                result = self._run_tool_call(call)
                self._history.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                })
        return "(the model kept calling tools without replying)"

    def _run_tool_call(self, call) -> Dict:
        # Feed errors back to the model so it can correct a bad script —
        # tool-call quality varies by OpenRouter provider routing.
        if call.function.name != "perform":
            return {"ok": False, "error": f"unknown tool {call.function.name!r}"}
        try:
            script = json.loads(call.function.arguments)["script"]
            return {"ok": True, "duration": self._perform(script)}
        except (json.JSONDecodeError, KeyError, TypeError, RuntimeError) as exc:
            return {"ok": False, "error": str(exc)}

    def _fallback_json_script(self, content: str) -> str:
        """Graceful degrade: some providers return JSON text, not tool calls."""
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "script" in data:
                self._perform(data["script"])
                return str(data.get("reply", ""))
        except (json.JSONDecodeError, RuntimeError):
            pass
        return content


def main() -> None:
    from openai import OpenAI

    api_key = (os.environ.get("AVATAR_LLM_API_KEY")
               or os.environ.get("OPENROUTER_API_KEY")
               or os.environ.get("OPENAI_API_KEY"))
    if not api_key:
        raise SystemExit("set AVATAR_LLM_API_KEY (or OPENROUTER_API_KEY)")
    llm = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("AVATAR_LLM_BASE_URL",
                                "https://openrouter.ai/api/v1"),
    )
    model = os.environ.get("AVATAR_LLM_MODEL", "deepseek/deepseek-chat")

    with AvatarClient() as avatar:
        brain = AgentBrain(
            llm, model, perform=avatar.perform,
            system_prompt=SYSTEM_PROMPT.format(
                emotions=", ".join(avatar.emotions())
            ),
        )
        print(f"Chat with your avatar ({model}; Ctrl-D to exit).")
        while True:
            try:
                user_text = input("you> ")
            except EOFError:
                break
            print(f"avatar> {brain.respond(user_text)}")
        avatar.reset()


if __name__ == "__main__":
    main()
