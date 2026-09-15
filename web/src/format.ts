export const DASH = '—';

export function num(v: number | null | undefined, digits?: number): string {
  if (v == null || Number.isNaN(v)) return DASH;
  const d = digits ?? (Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 1 : 2);
  return v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
}

export function gb(bytes: number | null | undefined): string {
  return bytes == null ? DASH : `${num(bytes / 1e9, 2)} GB`;
}

export function depthLabel(d: number): string {
  if (d === 0) return '0';
  return d % 1024 === 0 ? `${d / 1024}k` : d >= 1000 ? `${num(d / 1000, 1)}k` : String(d);
}

export function date(iso: string | null | undefined): string {
  return iso ? iso.slice(0, 16).replace('T', ' ') : DASH;
}

export function testLabel(m: { key: string; n_prompt: number; n_gen: number }): string {
  if (m.key === 'pp_tps') return `pp${m.n_prompt}`;
  if (m.key === 'tg_tps') return `tg${m.n_gen}`;
  return m.n_prompt ? `pp${m.n_prompt}` : m.n_gen ? `tg${m.n_gen}` : '';
}
