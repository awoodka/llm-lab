"""Self-test harness: one question, plus checks that the box is as closed as the lab asks for."""

import os
import socket
import subprocess

from boxproto import Channel

channel = Channel()


def list_tasks():
    return {"tasks": [{"id": "selftest/sum", "meta": {}}], "source": {"harness": "selftest"}}


def isolation() -> dict:
    """What a program the model wrote could reach from in here."""
    print("stray output that must not reach the channel")
    subprocess.run(["sh", "-c", "echo child output; cat"], check=False)  # the child sees /dev/null on stdin
    pid = os.fork()
    if pid == 0:  # a forked solution trying to speak for the box
        try:
            os.write(channel._out.fileno(), b'{"op": "result", "task": "forged", "attempt": 0, "passed": true}\n')
        finally:
            os._exit(0)
    os.waitpid(pid, 0)
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=2)
        network = True
    except OSError:
        network = False
    try:
        open("/box/written", "w").close()
        root_writable = True
    except OSError:
        root_writable = False
    return {"network": network, "root_writable": root_writable, "uid": os.getuid(), "tmp_writable": os.access("/tmp", os.W_OK)}


def run_task(task: str, attempt: int, options: dict) -> dict:
    body = {"messages": [{"role": "user", "content": "What is 2 + 2? Answer with the number only."}]}
    reply = channel.chat(body)
    text = reply["choices"][0]["message"].get("content") or ""
    return {
        "passed": "4" in text,
        "extracted": text.strip()[:20],
        "detail": isolation(),
        "transcript": [{"request": body, "reply": reply}],
    }


channel.serve(list_tasks, run_task)
