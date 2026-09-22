# History rewrite, 2026-09-22

This repository went public on 2026-09-22. Before that it held one homelab's deployment details: private network
names and addresses, filesystem paths on its hosts, an inbox, disk and container ids, and the names of other sites on
the same server. The history was rewritten once, with
[git-filter-repo](https://github.com/newren/git-filter-repo) 2.47.0 and a private replacement map, so that no commit
contains them. Placeholders took their place: `example-tailnet`, `100.64.0.1`, `you@example.com`, `/opt/llmlab`,
`other-app.example.com`, `<disk>`, `<ctid>`. The commit messages also lost their `Co-Authored-By` trailers. Nothing
else changed: every commit keeps its author, dates and message, and the final tree is byte for byte the one before the
rewrite.

Every commit id changed. Runs record the lab's version as `0.1.0+<commit>`, and runs published before the rewrite
carry the old ids, so this table maps them. The ones a published run records are marked.

| Date | Old | New | Commit |
|---|---|---|---|
| 2026-09-15 | `6e93045` | `fb656f2` | Baseline: lab CLI and showcase site as deployed on 2026-09-15 |
| 2026-09-15 | `4188c81` | `1c8d8b0` | Refuse GPU work above the power cap; roll back failed promotes |
| 2026-09-15 | `16df55c` (on the site) | `4d092e3` | Add Qwen3.8 27B Q4_K_M (bartowski) with 32k and 64k configs |
| 2026-09-15 | `c570a52` | `d6a6835` | Add lab gpu run and a process-group server helper |
| 2026-09-15 | `5300918` | `501becb` | Site foundations for capability evals (schema v2, scoring, tiers) |
| 2026-09-15 | `e70c2cb` | `71230f5` | Mirror contract v2 in the lab publish schema |
| 2026-09-15 | `7723386` | `892a987` | Capability pages: scoreboard, capability table and per-config detail |
| 2026-09-16 | `c56ee38` | `92c1db9` | Lead the homepage with the work, not the tokens per second |
| 2026-09-16 | `f77151c` | `fb709c2` | Allowance and the proxy that enforces it |
| 2026-09-16 | `29aef50` | `f74b23d` | lab eval: the quick-tier runner, with AIME 2025 as its first benchmark |
| 2026-09-16 | `efb1036` (on the site) | `8caa0aa` | Keep eval replies, count Gemma's BOS, and exclude tasks from the record |
| 2026-09-16 | `3fc62d0` | `25c6193` | Methodology: speed runs have no chat template; drop a premature credit |
| 2026-09-16 | `c1bb695` | `bcaf344` | Sandboxed harnesses on web, and one time limit per task |
| 2026-09-16 | `3aa4157` | `f1aa9bc` | BFCL in the sandbox, graded by its own checkers |
| 2026-09-16 | `6733613` | `3cae640` | LiveCodeBench in the sandbox; runs that span sittings stay one measurement |
| 2026-09-16 | `bfb7640` | `9d40757` | BFCL prompting mode for chat templates without tools |
| 2026-09-16 | `edfcd68` | `5fdbc19` | GPQA Diamond driver, ready for when the dataset is unlocked |
| 2026-09-16 | `a29cd5a` | `68efddd` | Methodology: who grades what, BFCL modes, shared task limits |
| 2026-09-16 | `d7a6194` | `0adafd2` | lab eval says what a tier still needs instead of offering to publish it |
| 2026-09-17 | `ba2d048` (on the site) | `ec89706` | GPQA login hint names the CLI that ships with the lab |
| 2026-09-17 | `9e7b5d0` | `b97b485` | Pin GPQA Diamond at 633f5ee (ids only) |
| 2026-09-17 | `1a3f1cd` | `011aaa8` | vLLM as a lab engine: the syv-ai launcher, pinned and checked |
| 2026-09-17 | `b18902a` | `249e953` | Site: chat benchmark on every page, vLLM alongside llama.cpp |
| 2026-09-17 | `795a2d1` (on the site) | `d386240` | One chat benchmark for every engine, over the API |
| 2026-09-17 | `ecfbd0f` | `8c717ae` | Tell the launcher where CUDA lives |
| 2026-09-17 | `e374dbb` | `1448588` | Document the vLLM engine and the two benchmark methods |
| 2026-09-17 | `af992eb` | `3603485` | Evals speak each engine's dialect, so vLLM can be scored |
| 2026-09-17 | `67a54c2` | `c98863f` | Say which mode a tool benchmark ran in |
| 2026-09-17 | `80d380e` (on the site) | `13efe02` | A run that dies no longer takes chat with it |
| 2026-09-22 | `7435234` | `ff14be8` | Evals keep vLLM's thinking and count it; lab runs thinking and markers |
| 2026-09-22 | `47747f3` | `db2fa35` | A config's pinned commit picks its own launcher checkout |
| 2026-09-22 | `17be23f` | `f9fb77a` | Deploy settings come from the app's .env |
| 2026-09-22 | `73074d7` | `f2c1506` | A public README, the Apache-2.0 license and a NOTICE |
| 2026-09-22 | `672461d` | `686e33d` | Nothing tracked or published points into the tailnet |
