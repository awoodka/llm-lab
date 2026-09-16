"""The sandbox channel. The last test runs the real selftest image on the sandbox host: LAB_SANDBOX_IT=1."""

import os
import shlex

import pytest

from lab.evals import sandbox
from lab.evals.sandbox import Box, context_hash, ensure_image, remote


def test_the_remote_command_survives_the_remote_shell():
    argv = ["python", "-c", "import sys; print('a b')", "$HOME", "x;y"]
    cmd = remote("alex@web", argv)
    assert cmd[:2] == ["ssh", "-o"] and cmd[-2] == "alex@web"
    assert shlex.split(cmd[-1]) == argv, "one quoted string, so the remote shell hands docker the same argv"


def test_an_image_tag_follows_its_build_context(tmp_path, monkeypatch):
    for part in ("common", "demo"):
        (tmp_path / part).mkdir()
    (tmp_path / "common" / "boxproto.py").write_text("x = 1\n")
    (tmp_path / "demo" / "Dockerfile").write_text("FROM scratch\n")
    monkeypatch.setattr(sandbox, "SANDBOX_DIR", tmp_path)
    before = context_hash("demo")
    assert context_hash("demo") == before
    (tmp_path / "common" / "boxproto.py").write_text("x = 2\n")
    assert context_hash("demo") != before, "a change to the shared channel code is a new image too"
    with pytest.raises(sandbox.SandboxError):
        context_hash("missing")


@pytest.mark.skipif(not os.environ.get("LAB_SANDBOX_IT"), reason="needs the sandbox host; set LAB_SANDBOX_IT=1")
def test_the_selftest_box_answers_through_the_channel_and_is_closed():
    def chat(body):
        return 200, {"choices": [{"message": {"role": "assistant", "content": "4"}, "finish_reason": "stop"}]}

    with Box(ensure_image("selftest")) as box:
        assert [t["id"] for t in box.list_tasks()["tasks"]] == ["selftest/sum"]
        result = box.run("selftest/sum", 1, chat)
    assert result["passed"]
    assert result["detail"] == {"network": False, "root_writable": False, "uid": 65534, "tmp_writable": True}
