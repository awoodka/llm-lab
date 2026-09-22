"""The repository is public: no tracked file may name the lab's private network, an inbox, or a host's disks.

Real values live in gitignored files (lab/settings.yaml, web/deploy/local.env, the app's .env on the web host),
and the examples use placeholders: example-tailnet, 100.64.0.1, example.com. scripts/prepush-check.sh does the
same for the whole history, with the real literals kept in private files outside the repository.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# Placeholders the docs and tests may use: the start of Tailscale's range (100.64.0.0/10) and its first address.
PLACEHOLDER_ADDRESSES = {"100.64.0.0", "100.64.0.1"}

PATTERNS = {
    "a tailnet name": re.compile(r"(?<![A-Za-z0-9-])(?!example-tailnet\.)[A-Za-z0-9-]+\.ts\.net\b"),
    "a tailnet address": re.compile(r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b"),
    "a gmail address": re.compile(r"@gmail\.com\b"),
    "a host's disk": re.compile(r"/dev/nvm[e]"),  # a class, so the pattern does not match itself
    "a Proxmox container command": re.compile(r"\bpct (?:set|reboot)\b"),
}


def leaks_in(text: str) -> list[tuple[str, re.Match]]:
    return [
        (what, m)
        for what, pattern in PATTERNS.items()
        for m in pattern.finditer(text)
        if m.group() not in PLACEHOLDER_ADDRESSES
    ]


def tracked_files() -> list[Path]:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"], capture_output=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [REPO / name for name in out.decode().split("\0") if name]


def test_no_tracked_file_leaks_private_infrastructure():
    found = []
    for path in tracked_files():
        if not path.is_file():  # deleted in the working tree, not yet committed
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        for what, m in leaks_in(text):
            found.append(f"{path.relative_to(REPO)}:{text.count(chr(10), 0, m.start()) + 1}: {what} ({m.group()!r})")
    assert not found, "\n".join(found)


# The samples are split in two so that this file passes its own scan.
@pytest.mark.parametrize(
    "text",
    ["https://ai.tailabc123" + ".ts.net:8443", "100.101" + ".102.103", "100.127" + ".255.254", "someone@" + "gmail.com",
     "wipefs -a /dev/" + "nvme1n1", "pct " + "set 101 -mp0 x", "pct " + "reboot 101"],
)
def test_the_patterns_catch_what_they_are_for(text):
    assert leaks_in(text), text


@pytest.mark.parametrize(
    "text",
    ["https://web.example-tailnet.ts.net", "AI_IP=100.64.0.1", "Tailscale's range (100.64.0.0/10)", "the homepage mentions *ts.net*",
     "100.63.1.1 and 100.128.0.1", "you@example.com", "/mnt/nvme/models", "print(pct)"],
)
def test_the_placeholders_pass(text):
    assert not leaks_in(text), text
