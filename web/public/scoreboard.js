// Draws the capability scoreboard from the JSON spec in <script id="...">, into <div class="board" data-scoreboard="id">.
// Ported from the approved mockup. Colors come from CSS tokens, so light and dark are selected palettes.
// The server-rendered capability table below it carries the same numbers for readers without JavaScript.
(() => {
  const NS = 'http://www.w3.org/2000/svg';
  const f1 = (v) => v.toFixed(1);

  function el(tag, attrs, parent) {
    const node = document.createElementNS(NS, tag);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }
  function label(parent, x, y, text, cls, anchor) {
    const t = el('text', { x, y, class: cls, 'text-anchor': anchor || 'start' }, parent);
    t.textContent = text;
    return t;
  }
  function tspan(parent, text, cls) {
    const s = el('tspan', { class: cls }, parent);
    s.textContent = text;
    return s;
  }
  function marker(parent, kind, cx, cy, cls) {
    if (kind === 'circle') return el('circle', { cx, cy, r: 5, class: cls }, parent);
    const s = 6.5;
    return el('path', { d: `M${cx} ${cy - s}L${cx + s} ${cy}L${cx} ${cy + s}L${cx - s} ${cy}Z`, class: cls }, parent);
  }
  function barPath(x, y, w, h) {
    const r = Math.max(0, Math.min(4, w, h / 2));
    return `M${x} ${y}H${x + w - r}Q${x + w} ${y} ${x + w} ${y + r}V${y + h - r}Q${x + w} ${y + h} ${x + w - r} ${y + h}H${x}Z`;
  }

  function setup(board) {
    const spec = JSON.parse(document.getElementById(board.dataset.scoreboard).textContent);
    const { categories: CATS, benchmarks: BENCHES, groups: GROUPS, speedLabel } = spec;
    const rows = GROUPS.flatMap((g) => g.rows);
    if (!rows.length) return;

    const tip = document.createElement('div');
    tip.className = 'tip';
    tip.hidden = true;
    board.append(tip);

    // -- tooltip ---------------------------------------------------------------------------
    function tipTitle(text) {
      const d = document.createElement('div');
      d.className = 'tip-title';
      d.textContent = text;
      tip.append(d);
    }
    function tipRow(value, text, cat) {
      const row = document.createElement('div');
      row.className = 'tip-row';
      const key = document.createElement('span');
      key.className = 'tip-key';
      if (cat) key.style.background = `var(--cat-${cat})`;
      const v = document.createElement('span');
      v.className = 'tip-value';
      v.textContent = value;
      const l = document.createElement('span');
      l.className = 'tip-label';
      l.textContent = text;
      row.append(key, v, l);
      tip.append(row);
    }
    function tipNote(text) {
      const d = document.createElement('div');
      d.className = 'tip-note';
      d.textContent = text;
      tip.append(d);
    }
    function place(evt, anchor) {
      const box = board.getBoundingClientRect();
      let x;
      let y;
      if (evt) {
        x = evt.clientX - box.left;
        y = evt.clientY - box.top;
      } else {
        const r = anchor.getBoundingClientRect();
        x = Math.min(r.left - box.left + 260, box.width * 0.55);
        y = r.top - box.top + r.height / 2;
      }
      const w = tip.offsetWidth;
      const h = tip.offsetHeight;
      let left = x + 16;
      let top = y + 16;
      if (left + w > box.width) left = Math.max(0, x - w - 16);
      if (top + h > box.height) top = Math.max(0, y - h - 16);
      tip.style.left = `${left}px`;
      tip.style.top = `${top}px`;
    }
    function show(fill, evt, anchor) {
      tip.replaceChildren();
      fill();
      tip.hidden = false;
      place(evt, anchor);
    }
    const hide = () => { tip.hidden = true; };
    function hover(node, fill) {
      node.addEventListener('pointerenter', (e) => show(fill, e));
      node.addEventListener('pointermove', (e) => { if (!tip.hidden) place(e); });
      node.addEventListener('pointerleave', hide);
    }

    const heading = (r) => `${r.name} · ${r.config}`;
    const benchTip = (r, b) => () => {
      const s = r.scores[b.key];
      tipTitle(heading(r));
      tipRow(`${f1(s.value)}% ± ${f1(s.stderr)}`, b.name, b.category);
      tipNote(`${CATS.find((c) => c.key === b.category).name} · ${b.tier} tier · ${b.size}`);
    };
    const scoreTip = (r) => () => {
      tipTitle(heading(r));
      const s = r.score ?? r.quick;
      tipRow(r.score ? `${f1(s.value)} ± ${f1(s.stderr)}` : f1(s.value), r.score ? `score · rank ${r.rankLabel}` : 'quick-tier score');
      s.categories.forEach((c) => tipRow(f1(c.mean), `${c.name} · ${Math.round(c.weight * 100)}%`, c.key));
      if (r.tieNote) tipNote(r.tieNote);
      if (!r.score) tipNote('Quick tier only: the deep-tier benchmarks have not been run for this config.');
    };
    const speedTip = (r) => () => {
      tipTitle(heading(r));
      if (r.speed != null) tipRow(`${f1(r.speed)} t/s`, speedLabel);
      if (r.promptSpeed != null) tipRow(`${Math.round(r.promptSpeed).toLocaleString('en-US')} t/s`, 'prompt reading');
      if (r.allowance != null) tipRow(r.allowanceText, 'thinking allowance, 5-minute task');
      if (r.issuesPerNight != null) tipRow(String(r.issuesPerNight), 'issues fixed per night');
      if (r.vramGb != null) tipRow(`${f1(r.vramGb)} GB`, 'peak VRAM');
    };

    const rowAria = (r) => [
      r.score ? `Rank ${r.rankLabel}: ${r.name}, ${r.config}. Score ${f1(r.score.value)}.` : `Not ranked: ${r.name}, ${r.config}. Quick-tier score ${f1(r.quick.value)}.`,
      `${BENCHES.map((b) => (r.scores[b.key] ? `${b.name} ${f1(r.scores[b.key].value)}%` : `${b.name} not run`)).join(', ')}.`,
      r.speed != null ? `${f1(r.speed)} tokens per second ${speedLabel}.` : '',
    ].join(' ');

    function render() {
      const width = Math.round(board.clientWidth);
      if (!width) return;
      const compact = width < 820;
      board.querySelector('svg')?.remove();
      hide();

      let L;
      if (compact) {
        L = { capX0: 8, capX1: width - 8, rowH: 108, tracks: [66, 82, 98] };
      } else {
        const labelW = 250;
        const scoreW = 70;
        const speedW = Math.max(170, Math.min(240, Math.round(width * 0.2)));
        const spX0 = width - speedW;
        L = { labelW, scoreW, capX0: labelW + scoreW + 24, capX1: spX0 - 40, spX0, spX1: width - 48, rowH: 68, tracks: [18, 34, 50] };
      }
      const cx = (v) => L.capX0 + (v / 100) * (L.capX1 - L.capX0);
      const speeds = rows.map((r) => r.speed).filter((v) => v != null);
      const speedMax = Math.max(50, Math.ceil(Math.max(...speeds, 0) / 50) * 50);
      const sx = (v) => (v / speedMax) * (L.spX1 - L.spX0);
      const capTicks = [0, 25, 50, 75, 100];
      const spTicks = [];
      for (let v = 0; v <= speedMax; v += 50) spTicks.push(v);

      const svg = el('svg', { width, 'aria-label': 'Every config ranked for coding work. The table below lists the same numbers.' });
      const gridLayer = el('g', {}, svg);
      const rowLayer = el('g', {}, svg);

      if (compact) {
        label(svg, L.capX0, 14, 'Capability · % solved', 't-panel');
      } else {
        label(svg, 26, 14, 'Model', 't-panel');
        label(svg, L.labelW + L.scoreW - 4, 14, 'Score', 't-panel', 'end');
        label(svg, L.capX0, 14, 'Capability · % solved', 't-panel');
        if (speeds.length) label(svg, L.spX0, 14, speedLabel, 't-panel');
      }

      function drawRow(r, top, last) {
        const g = el('g', { class: 'row', tabindex: '0', 'aria-label': rowAria(r) }, rowLayer);
        el('rect', { x: -6, y: top + 1, width: width + 12, height: L.rowH - 2, rx: 6, class: 'row-bg' }, g);
        if (!last) el('line', { x1: 0, x2: width, y1: top + L.rowH, y2: top + L.rowH, class: 'g-line' }, gridLayer);
        const ty = (k) => top + L.tracks[k];

        if (compact) {
          label(g, 0, top + 20, r.rankLabel, 't-rank');
          label(g, 26, top + 20, r.name, 't-name');
          const s = el('text', { x: width, y: top + 20, 'text-anchor': 'end' }, g);
          if (r.score) {
            tspan(s, f1(r.score.value), 't-score');
            tspan(s, ` ±${f1(r.score.stderr)}`, 't-se');
          } else {
            tspan(s, `quick ${f1(r.quick.value)}`, 't-se');
          }
          label(g, 26, top + 38, r.speed != null ? `${r.config} · ${f1(r.speed)} t/s` : r.config, 't-cfg');
          if (!r.score) label(g, 26, top + 54, 'Quick tier only', 't-flag');
        } else {
          const right = L.labelW + L.scoreW - 4;
          label(g, 0, top + 26, r.rankLabel, 't-rank');
          label(g, 26, top + 26, r.name, 't-name');
          label(g, 26, top + 44, r.config, 't-cfg');
          if (!r.score) label(g, 26, top + 60, 'Quick tier only', 't-flag');
          if (r.score) {
            label(g, right, top + 30, f1(r.score.value), 't-score', 'end');
            label(g, right, top + 47, `± ${f1(r.score.stderr)}`, 't-se', 'end');
          } else {
            label(g, right, top + 30, '—', 't-score is-none', 'end');
            label(g, right, top + 47, `quick ${f1(r.quick.value)}`, 't-se', 'end');
          }
        }

        // Overall score: a tick with its ± band behind the benchmark marks.
        if (r.score) {
          const s = r.score;
          const y0 = ty(0) - 9;
          const h = ty(2) - ty(0) + 18;
          const x0 = cx(Math.max(0, s.value - s.stderr));
          const x1 = cx(Math.min(100, s.value + s.stderr));
          const m = el('g', { class: 'mark' }, g);
          el('rect', { x: x0, y: y0, width: x1 - x0, height: h, rx: 2, class: 'ci' }, m);
          el('rect', { x: cx(s.value) - 1.5, y: y0, width: 3, height: h, rx: 1.5, class: 'tick' }, m);
          const hx0 = Math.min(x0, cx(s.value) - 8);
          const hx1 = Math.max(x1, cx(s.value) + 8);
          el('rect', { x: hx0, y: y0, width: hx1 - hx0, height: h, class: 'hit' }, m);
          hover(m, scoreTip(r));
        }

        CATS.forEach((c, k) => {
          const bs = BENCHES.filter((b) => b.category === c.key && r.scores[b.key]);
          if (bs.length === 2) {
            el('line', { x1: cx(r.scores[bs[0].key].value), x2: cx(r.scores[bs[1].key].value), y1: ty(k), y2: ty(k), class: `link l-${c.key}` }, g);
          }
          bs.forEach((b) => {
            const x = cx(r.scores[b.key].value);
            const m = el('g', { class: 'mark' }, g);
            marker(m, b.shape, x, ty(k), `dot c-${c.key}`);
            el('circle', { cx: x, cy: ty(k), r: 12, class: 'hit' }, m);
            hover(m, benchTip(r, b));
          });
        });

        if (!compact && r.speed != null) {
          const w = sx(r.speed);
          const m = el('g', { class: 'mark' }, g);
          el('path', { d: barPath(L.spX0, ty(1) - 6, w, 12), class: 'bar' }, m);
          label(m, L.spX0 + w + 6, ty(1) + 4, f1(r.speed), 't-bar');
          el('rect', { x: L.spX0, y: ty(0) - 9, width: width - L.spX0, height: ty(2) - ty(0) + 18, class: 'hit' }, m);
          hover(m, speedTip(r));
        }

        g.addEventListener('focus', () => { if (g.matches(':focus-visible')) show(scoreTip(r), null, g); });
        g.addEventListener('blur', hide);
      }

      let y = 26;
      for (const group of GROUPS) {
        if (!group.rows.length) continue;
        label(rowLayer, 0, y + 16, group.title, 't-group');
        const top0 = y + 26;
        group.rows.forEach((r, i) => drawRow(r, top0 + i * L.rowH, i === group.rows.length - 1));
        const bottom = top0 + group.rows.length * L.rowH;
        capTicks.forEach((v) => el('line', { x1: cx(v), x2: cx(v), y1: top0, y2: bottom, class: 'g-line' }, gridLayer));
        if (!compact && speeds.length) {
          spTicks.forEach((v) => {
            const x = L.spX0 + sx(v);
            el('line', { x1: x, x2: x, y1: top0, y2: bottom, class: v === 0 ? 'g-axis' : 'g-line' }, gridLayer);
          });
        }
        y = bottom + 8;
      }

      const axisY = y + 8;
      capTicks.forEach((v) => label(svg, cx(v), axisY, String(v), 't-axis', 'middle'));
      if (!compact && speeds.length) spTicks.forEach((v) => label(svg, L.spX0 + sx(v), axisY, String(v), 't-axis', 'middle'));
      const height = axisY + 8;
      svg.setAttribute('height', height);
      svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
      board.prepend(svg);
    }

    let lastWidth = 0;
    const rerender = () => {
      const w = Math.round(board.clientWidth);
      if (w && w !== lastWidth) {
        lastWidth = w;
        render();
      }
    };
    if ('ResizeObserver' in window) new ResizeObserver(rerender).observe(board);
    else window.addEventListener('resize', rerender);
    rerender();
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { lastWidth = 0; rerender(); });
  }

  document.querySelectorAll('[data-scoreboard]').forEach(setup);
})();
