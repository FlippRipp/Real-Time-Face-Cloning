#!/usr/bin/env python3
"""An LLM-puppeted avatar: Claude picks the avatar's emotion, gaze, and
speech, and performs them through the control channel.

This is the pattern for hooking any AI agent to the avatar:

1. Ask the model for a reply *plus* stage directions (as JSON).
2. Feed the stage directions to AvatarClient.
3. Pipe the reply text through speak_text (or a real TTS with phoneme
   timings via speak_visemes for accurate lipsync).

Requires the `anthropic` package and an ANTHROPIC_API_KEY. Start the avatar
first: python main.py --mock --driver procedural
"""

import json
import time

from tha_wrapper.control import AvatarClient

SYSTEM_PROMPT = """\
You are a VTuber avatar. Reply to the user in one or two short sentences.
Respond ONLY with JSON: {"reply": str, "emotion": str, "intensity": float,
"look": [x, y]} where emotion is one of: neutral, happy, sad, angry,
surprised, amused, smug, tired, confused; and look x/y are in [-1, 1].
"""


def perform(avatar: AvatarClient, directions: dict) -> None:
    avatar.emotion(
        directions.get("emotion", "neutral"),
        float(directions.get("intensity", 1.0)),
    )
    look = directions.get("look") or [0.0, 0.0]
    avatar.look(float(look[0]), float(look[1]), head_follow=0.4)
    duration = avatar.speak_text(directions.get("reply", ""))
    time.sleep(duration)


def main() -> None:
    import anthropic

    client = anthropic.Anthropic()
    history = []
    with AvatarClient() as avatar:
        print("Chat with your avatar (Ctrl-D to exit).")
        while True:
            try:
                user_text = input("you> ")
            except EOFError:
                break
            history.append({"role": "user", "content": user_text})
            response = client.messages.create(
                model="claude-sonnet-5",
                max_tokens=300,
                system=SYSTEM_PROMPT,
                messages=history,
            )
            raw = response.content[0].text
            history.append({"role": "assistant", "content": raw})
            try:
                directions = json.loads(raw)
            except json.JSONDecodeError:
                directions = {"reply": raw, "emotion": "neutral"}
            print(f"avatar[{directions.get('emotion')}]> "
                  f"{directions.get('reply')}")
            perform(avatar, directions)
        avatar.reset()


if __name__ == "__main__":
    main()
