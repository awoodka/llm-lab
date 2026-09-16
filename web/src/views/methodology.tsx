import { num } from '../format.ts';

/**
 * "How it's measured". Keep in step with the lab code: lab/src/lab/bench/native_llamacpp.py (power per test),
 * lab/src/lab/telemetry.py (sampling, throttle bits), lab/src/lab/gpu/hosted.py (clean start),
 * lab/src/lab/catalog.py (bench defaults, config hash).
 */
export const Methodology = (props: { hardware: Record<string, any>[]; builds: Record<string, any>[] }) => (
  <article class="prose">
    <p class="crumb"><a href="/benchmarks">← Benchmarks</a></p>
    <h1>How it's measured</h1>
    <p class="lead">
      Every number on this site comes from a scripted run on one machine. This page explains what each run does, what the
      numbers mean, and what they don't cover.
    </p>

    <h2>The machine</h2>
    {props.hardware.length > 0 ? (
      <ul>
        {props.hardware.map((h) => (
          <li>
            {h.gpu_name} with {Math.round(h.gpu_vram_mb / 1024)} GB of VRAM at a {num(h.power_limit_w, 0)} W power limit (driver {h.driver},
            CUDA {h.cuda}); {h.cpu_model} with {h.cpu_threads_visible} threads; {Math.round(h.ram_mb / 1024)} GB of RAM.
          </li>
        ))}
        {props.builds.length > 0 && <li>Engine builds: {props.builds.map((b) => `${b.engine}@${b.commit_sha ?? '?'}`).join(', ')}.</li>}
      </ul>
    ) : (
      <p>Hardware details appear here once results are published.</p>
    )}

    <h2>Speed runs</h2>
    <p>
      Speed comes from llama.cpp's own benchmark tool, <code>llama-bench</code>, running directly on the GPU with no server in
      the loop. By default each config is tested with:
    </p>
    <ul>
      <li><strong>Prompt processing:</strong> a 512-token prompt.</li>
      <li><strong>Generation:</strong> 128 new tokens.</li>
      <li>
        <strong>Context depth:</strong> 0, 4k and 16k tokens already in the cache, to show how speed falls as a conversation
        grows. Depths that don't fit a config's context length are skipped.
      </li>
      <li><strong>Repetitions:</strong> 5 of each test, with a 3-second pause between tests. Tables show the mean and standard deviation.</li>
    </ul>

    <h2>Every setting is explicit</h2>
    <p>
      Each config's exact launch command is on its model page. Settings that an engine would otherwise pick for itself are
      always pinned: <code>--fit off</code>, parallel slots (<code>-np</code>), <code>--cache-ram 0</code>, flash attention,
      KV-cache precision, batch and micro-batch size, CPU threads, MoE expert offload (<code>-ncmoe</code>) and how weights
      are loaded.
    </p>
    <p>Configs are hash-locked: changing any setting creates a new config instead of quietly rewriting published results.</p>

    <h2>A clean GPU for every run</h2>
    <p>
      Before a run starts, the lab takes an exclusive lock on the GPU, stops the chat model this machine hosts, and waits
      until less than 300 MiB of VRAM is in use and the GPU has cooled to 45 °C or below. The chat model restarts when the
      run ends, even if the run fails.
    </p>

    <h2>Power and efficiency</h2>
    <ul>
      <li>GPU telemetry (VRAM, power, temperature, utilization and throttle reasons) is sampled every 100 ms through NVIDIA's NVML.</li>
      <li><strong>Power</strong> for a test is the mean GPU power over that test's samples where utilization was at least 50%.</li>
      <li>
        <strong>Tokens per joule</strong> is tokens per second divided by that power. Only GPU power counts: CPU power isn't
        readable from inside the container.
      </li>
      <li>
        <strong>Peak VRAM</strong> reflects llama-bench's cache sizing for its tests (prompt, generated tokens and depth), not a
        server holding a full context window.
      </li>
    </ul>

    <h2>Capability evals</h2>
    <p>
      Speed alone doesn't say whether a model can do the work, so each config is also scored on what it solves. Every
      benchmark runs against the config's own server, with its own settings, and reports the percent of tasks solved with
      a standard error.
    </p>
    <ul>
      <li>
        <strong>Quick tier</strong>, run for every config: LiveCodeBench (100 problems), BFCL (400 cases), GPQA Diamond
        (198 questions) and AIME 2025 (30 problems, 4 attempts each). Each task gets 5 minutes.
      </li>
      <li>
        <strong>Deep tier</strong>, run only for configs worth the overnight time: SWE-bench Verified (30 issues, solved
        by an agent) and Terminal-Bench (30 tasks). Each task gets 20 minutes and runs alone.
      </li>
      <li>Only configs with all six benchmarks are ranked. The rest are listed by their quick-tier score.</li>
      <li>
        Each config is evaluated exactly as it serves chat, including thinking where it has it. Thinking tokens come out
        of the allowance, so reasoning is paid for rather than free. Speed runs don't involve thinking at all: llama-bench
        times fixed token counts without a chat template.
      </li>
      <li>
        AIME 2025 and GPQA Diamond are graded by the lab: the boxed integer, and the last "Answer: X" line of the answer.
        LiveCodeBench and BFCL run their own published code for prompts and grading (LiveCodeBench at a pinned commit,
        bfcl-eval 2026.3.23); the lab only passes the conversation to the model and back.
      </li>
      <li>
        LiveCodeBench uses the 100 newest problems of its v6 release, from 15 February to 6 April 2025. BFCL uses 80 cases
        from each of five categories: simple, multiple, parallel, parallel-multiple and multi-turn.
      </li>
      <li>
        BFCL calls tools natively when a model's chat template supports them. Otherwise it uses BFCL's prompting mode, which
        lists the functions in the system prompt, and the result says which mode ran.
      </li>
      <li>
        Code a model writes during a benchmark runs in a locked-down container with no network access, separate from the
        model server.
      </li>
    </ul>

    <h2>The thinking allowance</h2>
    <p>
      Speed matters here because it buys reasoning, not because a fast answer is worth more. Each task gives a config a
      token budget instead of a stopwatch: what it could generate within the time limit at its own measured speed, after
      reading the prompt.
    </p>
    <ul>
      <li>Allowance = (time limit − time to read the prompt) × generation speed at that prompt's depth, capped by the context left.</li>
      <li>Both figures come from the config's published speed run, so the budget is fixed before the benchmark starts and never drifts with load.</li>
      <li>Thinking tokens count against it. An answer that doesn't finish inside its allowance is wrong.</li>
      <li>
        A task that takes several requests, such as a multi-turn BFCL case, shares one time limit. Each request is charged
        for the prompt tokens the server hadn't already cached and for its output, at the measured speeds, and the next
        request gets what's left.
      </li>
      <li>A faster config gets more room to think in the same 5 minutes; a model that finishes early gains nothing from extra speed.</li>
    </ul>

    <h2>The coding-work score</h2>
    <ul>
      <li>Coding counts 50%, agents and tools 30%, reasoning 20%.</li>
      <li>
        A category's score is the mean of its benchmarks, and the total combines the three as a weighted geometric mean, so
        being weak in one area pulls the score down rather than averaging out.
      </li>
      <li>The ± figure is one standard error, carried through from each benchmark's own error.</li>
      <li>Configs whose scores are within one combined standard error share a rank, shown as "=2".</li>
      <li>Issues fixed per night = 8 hours ÷ the average time per SWE-bench issue × the share it resolved.</li>
    </ul>

    <h2>Throttling</h2>
    <p>
      A run is flagged as throttled if the GPU reports hardware slowdown, thermal slowdown or a hardware power brake at any
      point. Running into the card's configured power limit is expected and doesn't count. Throttled runs stay visible with a
      tag, but they never feed the headline numbers.
    </p>

    <h2>Reproducibility</h2>
    <p>
      Every run records a hardware snapshot (GPU, driver, CUDA, power limit, CPU, RAM and kernel), the llama.cpp commit, and
      the exact command. Each run has its own page with everything that was published, including the benchmark settings and a
      telemetry summary.
    </p>

    <h2>What these numbers don't show</h2>
    <ul>
      <li>They describe this one machine, not every RTX 3090.</li>
      <li>Speed runs measure one request at a time with synthetic prompts, not multi-user serving throughput.</li>
      <li>
        The benchmarks are small and use fixed subsets: 30 tasks carry roughly ±9 points of error near 50%, which is what
        the error bars and shared ranks are for.
      </li>
      <li>Each config is benchmarked with one sampling setup, the one it serves chat with. Another setting might score differently.</li>
      <li>
        Models may have seen these public benchmarks during training, which would flatter them. The LiveCodeBench problems
        date from early 2025, so models trained after that are the most likely to have seen them.
      </li>
      <li>Only configs worth the overnight time get the deep tier, so most are ranked on the quick tier alone.</li>
      <li>Output quality (perplexity, KL divergence) isn't measured yet.</li>
    </ul>
  </article>
);
