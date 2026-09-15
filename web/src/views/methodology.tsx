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
      <li>They measure one request at a time with synthetic prompts, not multi-user serving throughput or end-to-end chat latency.</li>
      <li>Output quality (perplexity, KL divergence) and task evals aren't measured yet.</li>
    </ul>
  </article>
);
