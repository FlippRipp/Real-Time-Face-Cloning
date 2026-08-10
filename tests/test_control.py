import random

import pytest

from tha_wrapper.control import AvatarClient, ControlServer
from tha_wrapper.drivers.procedural import ProceduralDriver


@pytest.fixture()
def server():
    driver = ProceduralDriver(auto_blink=False, breathing=False,
                              idle_sway=False, rng=random.Random(3))
    server = ControlServer(driver, host="127.0.0.1", port=0)
    server.start()
    server.port = server._server.server_address[1]
    yield server
    server.close()


@pytest.fixture()
def client(server):
    client = AvatarClient(port=server.port)
    yield client
    client.close()


def run(driver, seconds, dt=1 / 60):
    pose = driver.update(dt)
    for _ in range(int(seconds / dt)):
        pose = driver.update(dt)
    return pose


def test_emotion_roundtrip(server, client):
    client.emotion("happy", intensity=0.5, duration=0.05)
    pose = run(server.driver, 0.3)
    assert pose.get("eyebrow_happy_left") == pytest.approx(0.5, abs=1e-4)
    assert client.state()["emotion"] == "happy"


def test_pose_command(server, client):
    client.pose({"head_y": -0.4}, duration=0.0)
    assert run(server.driver, 0.05).get("head_y") == pytest.approx(-0.4)


def test_speak_text_returns_duration(server, client):
    duration = client.speak_text("hello world")
    assert duration > 0.2
    assert client.state()["talking"] is True


def test_look_and_blink(server, client):
    client.look(0.5, 0.5, duration=0.0)
    client.blink()
    peak = 0.0
    for _ in range(60):
        pose = server.driver.update(1 / 120)
        peak = max(peak, pose.get("eye_wink_left"))
    assert peak > 0.9
    assert pose.get("iris_rotation_y") == pytest.approx(0.5)


def test_bad_command_reports_error(server, client):
    with pytest.raises(RuntimeError, match="unknown command"):
        client.send("dance")
    with pytest.raises(RuntimeError, match="unknown emotion"):
        client.emotion("melancholy")
    # channel still alive afterwards
    assert client.emotions()


def test_idle_toggles(server, client):
    client.idle(auto_blink=True, sway=False, sway_amount=0.5)
    assert server.driver.auto_blink is True
    assert server.driver.idle_sway is False
    assert server.driver.idle_sway_amount == 0.5


def test_help_lists_commands(server, client):
    commands = client.send("help")["commands"]
    assert "emotion" in commands and "speak_visemes" in commands
