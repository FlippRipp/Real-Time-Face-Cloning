#!/usr/bin/env python3
"""Drive the avatar through a scripted performance over the control channel.

Start the avatar first (mock mode is fine for trying it out):

    python main.py --mock --driver procedural

then run this script from another terminal:

    python examples/scripted_performance.py
"""

import time

from tha_wrapper.control import AvatarClient


def main() -> None:
    with AvatarClient() as avatar:
        print("emotions available:", avatar.emotions())

        print("greeting...")
        avatar.emotion("happy")
        duration = avatar.speak_text("Hello everyone! It is so nice to see you.")
        time.sleep(duration + 0.3)

        print("looking around...")
        avatar.look(0.7, 0.2, duration=0.4, head_follow=0.6)
        time.sleep(1.0)
        avatar.look(-0.7, 0.1, duration=0.5, head_follow=0.6)
        time.sleep(1.0)
        avatar.look(0.0, 0.0, duration=0.4, head_follow=0.6)

        print("acting surprised...")
        avatar.emotion("surprised", duration=0.15)
        avatar.head(pitch=0.3, duration=0.15)
        time.sleep(1.2)

        print("precise viseme lipsync ('o-ha-yo-u')...")
        avatar.emotion("happy", intensity=0.7)
        avatar.head(pitch=0.0, duration=0.3)
        avatar.speak_visemes([
            ("oh", 0.18), ("aa", 0.22), ("oh", 0.18), ("ou", 0.28),
            ("sil", 0.2),
        ])
        time.sleep(1.4)

        print("smug wink...")
        avatar.emotion("smug")
        avatar.pose({"eye_wink_left": 1.0}, duration=0.12)
        time.sleep(0.8)
        avatar.pose({"eye_wink_left": 0.0}, duration=0.2)
        time.sleep(1.0)

        print("back to neutral")
        avatar.reset()
        print("final state:", avatar.state())


if __name__ == "__main__":
    main()
