// Renders every <canvas data-chart="id"> from the JSON spec in <script id="id">.
// Colors come from CSS tokens so light/dark are selected palettes, not a flip.
(() => {
  const charts = [];
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const fmt = (v) => {
    if (v == null || Number.isNaN(v)) return '—';
    const a = Math.abs(v);
    const d = a >= 100 ? 0 : a >= 10 ? 1 : 2;
    return v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
  };

  const withAlpha = (hex, alpha) => {
    const h = hex.replace('#', '');
    const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
  };

  // Value at the tip of each horizontal bar, in text ink (never the series color).
  const tipLabels = {
    id: 'tipLabels',
    afterDatasetsDraw(chart) {
      const { ctx } = chart;
      ctx.save();
      ctx.fillStyle = css('--text-secondary');
      ctx.font = `12px ${css('--font')}`;
      ctx.textBaseline = 'middle';
      chart.getDatasetMeta(0).data.forEach((bar, i) => {
        const v = chart.data.datasets[0].data[i];
        if (v != null) ctx.fillText(fmt(v), bar.x + 6, bar.y);
      });
      ctx.restore();
    },
  };

  function build(canvas) {
    const spec = JSON.parse(document.getElementById(canvas.dataset.chart).textContent);
    const ink = css('--text-primary');
    const ink2 = css('--text-secondary');
    const muted = css('--text-muted');
    const grid = css('--grid');
    const axis = css('--axis');
    const surface = css('--surface');
    const color = (slot) => (slot === 'muted' ? css('--series-muted') : css(`--series-${slot}`));
    const font = { family: css('--font'), size: 12 };

    const scale = (title, extra = {}) => ({
      grid: { color: grid, drawTicks: false },
      border: { color: axis },
      ticks: { color: muted, font, padding: 6 },
      title: { display: !!title, text: title, color: muted, font },
      ...extra,
    });

    const tooltip = {
      backgroundColor: surface,
      titleColor: ink2,
      bodyColor: ink,
      borderColor: css('--border'),
      borderWidth: 1,
      padding: 10,
      usePointStyle: true,
      titleFont: font,
      bodyFont: { ...font, weight: '600' },
      callbacks: {
        label(ctx) {
          const s = spec.series[ctx.datasetIndex];
          const v = spec.type === 'bar-h' ? ctx.parsed.x : ctx.parsed.y;
          const sd = s.stddev?.[ctx.dataIndex];
          const value = `${fmt(v)} ${spec.unit}${sd != null ? ` ± ${fmt(sd)}` : ''}`;
          if (spec.type === 'scatter') return `${value}  ${ctx.raw.label ?? ''}`;
          return spec.series.length > 1 ? `${value}  ${s.label}` : value;
        },
        title(items) {
          if (spec.type === 'scatter') return `${fmt(items[0].parsed.x)} ${spec.xLabel ?? ''}`;
          if (spec.type === 'line' && !spec.categories) return `${fmt(items[0].parsed.x)} s`;
          return items[0].label;
        },
      },
    };

    const legend = {
      display: spec.series.length > 1,
      position: 'top',
      align: 'start',
      labels: { color: ink2, font, usePointStyle: true, pointStyle: spec.type === 'line' ? 'line' : 'rect', boxWidth: 18, padding: 14 },
    };

    const base = {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      layout: { padding: { right: spec.type === 'bar-h' ? 48 : 8 } },
      plugins: { legend, tooltip },
    };

    let config;
    if (spec.type === 'bar-h') {
      config = {
        type: 'bar',
        data: {
          labels: spec.categories,
          datasets: spec.series.map((s) => ({
            label: s.label,
            data: s.data,
            backgroundColor: color(s.slot),
            hoverBackgroundColor: withAlpha(color(s.slot), 0.8),
            borderRadius: { topRight: 4, bottomRight: 4 },
            borderSkipped: 'start',
            maxBarThickness: 24,
          })),
        },
        options: {
          ...base,
          indexAxis: 'y',
          scales: { x: scale(spec.xLabel, { beginAtZero: true }), y: { grid: { display: false }, border: { color: axis }, ticks: { color: ink2, font } } },
        },
        plugins: [tipLabels],
      };
    } else if (spec.type === 'line') {
      const categorical = !!spec.categories;
      config = {
        type: 'line',
        data: {
          labels: spec.categories,
          datasets: spec.series.map((s) => {
            const c = color(s.slot);
            const stroke = s.dim ? withAlpha(c, 0.45) : c;
            return {
              label: s.label,
              data: s.data,
              borderColor: stroke,
              backgroundColor: stroke,
              borderWidth: 2,
              borderJoinStyle: 'round',
              borderCapStyle: 'round',
              pointRadius: categorical ? 4 : 0,
              pointHoverRadius: 6,
              pointBorderColor: surface,
              pointBorderWidth: 2,
              pointHitRadius: 12,
              spanGaps: true,
              tension: 0,
            };
          }),
        },
        options: {
          ...base,
          interaction: { mode: categorical ? 'index' : 'nearest', axis: 'x', intersect: false },
          scales: {
            x: categorical ? scale(spec.xLabel) : scale(spec.xLabel, { type: 'linear' }),
            y: scale(spec.yLabel, { beginAtZero: true }),
          },
        },
      };
    } else {
      config = {
        type: 'scatter',
        data: {
          datasets: spec.series.map((s) => ({
            label: s.label,
            data: s.data,
            backgroundColor: color(s.slot),
            borderColor: surface,
            borderWidth: 2,
            pointRadius: 6,
            pointHoverRadius: 8,
            pointHitRadius: 14,
          })),
        },
        options: { ...base, scales: { x: scale(spec.xLabel, { beginAtZero: true }), y: scale(spec.yLabel, { beginAtZero: true }) } },
      };
    }

    const chart = new Chart(canvas, config);
    if (spec.links) {
      canvas.style.cursor = 'pointer';
      canvas.addEventListener('click', (evt) => {
        const hit = chart.getElementsAtEventForMode(evt, 'nearest', { intersect: true }, false)[0];
        const href = hit && spec.links[hit.index];
        if (href) window.location.href = href;
      });
    }
    return chart;
  }

  function renderAll() {
    if (typeof Chart === 'undefined') return;
    charts.splice(0).forEach((c) => c.destroy());
    document.querySelectorAll('canvas[data-chart]').forEach((canvas) => charts.push(build(canvas)));
  }

  renderAll();
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', renderAll);

  document.querySelectorAll('[data-copy]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const text = document.getElementById(btn.dataset.copy)?.textContent ?? '';
      try {
        await navigator.clipboard.writeText(text);
        btn.textContent = 'Copied';
      } catch {
        btn.textContent = 'Select & copy';
      }
      setTimeout(() => (btn.textContent = 'Copy'), 1500);
    });
  });
})();
