"""The container side of the lab's sandbox channel. The lab side, and the protocol, are in
lab/src/lab/evals/sandbox.py.

The protocol runs on private copies of stdin and stdout. Descriptors 0 and 1 are pointed at /dev/null
and stderr before any harness code runs, so a print, or a program the model wrote, can't write into the
channel or read from it by accident.
"""

import itertools
import json
import os
import sys
import traceback
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any


class ModelUnavailable(RuntimeError):
    """The lab couldn't get an answer from the model. That's the lab's failure, not the model's."""


class Channel:
    def __init__(self) -> None:
        self._in = os.fdopen(os.dup(0), "r", buffering=1, encoding="utf-8")
        self._out = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
        for fd in (self._in.fileno(), self._out.fileno()):
            os.set_inheritable(fd, False)
        devnull = os.open(os.devnull, os.O_RDONLY)
        os.dup2(devnull, 0)
        os.close(devnull)
        os.dup2(2, 1)
        sys.stdin = open(os.devnull)  # noqa: SIM115 - lives as long as the process
        sys.stdout = sys.stderr
        self._ids = itertools.count(1)

    def send(self, **msg: Any) -> None:
        self._out.write(json.dumps(msg, default=str) + "\n")
        self._out.flush()

    def recv(self) -> dict[str, Any] | None:
        line = self._in.readline()
        return json.loads(line) if line else None

    def log(self, message: str) -> None:
        self.send(op="log", message=message)

    def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        """One chat completion from the model, via the lab. Sampling and length are the lab's to set."""
        rid = next(self._ids)
        self.send(op="chat", id=rid, body=body)
        reply = self.recv()
        if reply is None:
            raise ModelUnavailable("the lab closed the channel")
        if reply.get("op") != "reply" or reply.get("id") != rid:
            raise ModelUnavailable(f"expected reply {rid}, got {str(reply)[:200]}")
        if reply["status"] >= 400:
            raise ModelUnavailable(f"model request failed with {reply['status']}: {str(reply['body'])[:300]}")
        return reply["body"]

    def serve(self, list_tasks: Callable[[], dict[str, Any]], run_task: Callable[[str, int], dict[str, Any]]) -> None:
        """Answer the lab until it closes the channel.

        `run_task` returns the verdict: passed, extracted, detail, transcript. Anything it raises is a
        harness failure, so a harness must catch what a bad answer can cause and grade it as wrong.
        """
        while (msg := self.recv()) is not None:
            op = msg.get("op")
            if op == "list":
                self.send(op="tasks", **list_tasks())
            elif op == "run":
                task, attempt = msg["task"], msg["attempt"]
                try:
                    verdict = run_task(task, attempt)
                except Exception:  # noqa: BLE001 - reported to the lab, which retries or excludes
                    self.send(op="failed", task=task, attempt=attempt, message=traceback.format_exc()[-3000:])
                    continue
                self.send(op="result", task=task, attempt=attempt, **verdict)
            else:
                self.send(op="failed", task=msg.get("task"), attempt=msg.get("attempt"), message=f"unknown op {op!r}")


def _jsonable(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop the SDK's NOT_GIVEN sentinels and anything else that isn't request data."""
    out = {}
    for k, v in kwargs.items():
        if v is None or type(v).__name__ in ("NotGiven", "Omit"):
            continue
        out[k] = v
    # Pydantic messages (a harness may put the SDK's own reply objects back into the history) are dumped
    # the way the SDK itself sends them.
    return json.loads(json.dumps(out, default=lambda o: o.model_dump(exclude_unset=True, mode="json") if hasattr(o, "model_dump") else str(o)))


class ChannelOpenAI:
    """Just enough of `openai.OpenAI` for a harness that calls `client.chat.completions.create(...)`.

    Returns the SDK's own `ChatCompletion`, so the harness reads the answer exactly as it would from
    the real client. Streaming isn't offered: no harness here needs it.
    """

    def __init__(self, channel: Channel, requests: list[dict[str, Any]] | None = None):
        self._channel = channel
        #: Every request and reply of the current task, for its transcript.
        self.requests = requests if requests is not None else []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any):
        from openai.types.chat import ChatCompletion

        if kwargs.get("stream"):
            raise NotImplementedError("the sandbox channel doesn't stream")
        body = _jsonable(kwargs)
        reply = self._channel.chat(body)
        self.requests.append({"request": body, "reply": reply})
        return ChatCompletion.model_validate(reply)

    def transcript(self) -> dict[str, Any]:
        """The task's conversation, once: every request re-sends the history, so keep the last one whole.

        `messages` is the last request's history plus its reply; `tools` is the first request's list;
        `requests` has each request's size and how it ended.
        """
        if not self.requests:
            return {"messages": [], "requests": []}
        first, last = self.requests[0], self.requests[-1]
        final = (last["reply"].get("choices") or [{}])[0].get("message")
        return {
            "tools": first["request"].get("tools"),
            "messages": last["request"].get("messages", []) + ([final] if final else []),
            "requests": [
                {
                    "messages": len(r["request"].get("messages", [])),
                    "usage": r["reply"].get("usage"),
                    "finish_reason": (r["reply"].get("choices") or [{}])[0].get("finish_reason"),
                }
                for r in self.requests
            ],
        }
