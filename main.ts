// Roblox Top Trending forecaster (Node 22, standard library only, deterministic, no network).
//
// It reads the chart's history (history.ndjson) and the task (task.json) from $MAKEFUN_INPUT_DIR or /input, and writes
// forecast.json to $MAKEFUN_OUTPUT_DIR or /output: for each of the top 10 places, a probability for every game on the
// last chart, and a top-10 guess.
//
// The model, per game g on the last chart (rank r):
//   - position in a warped rank space, y = (10 + c) * ln(r + c): c small makes the top places tight and the lower places
//     loose, c large is close to plain ranks;
//   - drift: its velocity in that space over the last W hours (from the read nearest W hours back), shrunk by alpha and
//     projected h hours ahead, capped, never above #1;
//   - at an offset of h hours, place k gets weight exp(-|y(k) - y_g(h)| / s(h)) with s(h) = s0 + s1 * sqrt(h), normalised
//     over the games at that place, mixed with eps of a flat share over the chart, and (1 - nu) of the place in all: nu is
//     left unnamed for games that are not on the last chart yet;
//   - one forecast serves every slot of the window, so the per-hour forecasts are averaged over the window's offsets
//     (the log score's best single forecast for a mixture of hours).
// The parameters (c, s0, s1, eps, nu, alpha, W) are fitted by the program itself on the history: it backtests every
// recent read as an origin, forecasts the reads that followed it within the same horizon, scores them with the contest's
// own log score, and picks the grid values by coordinate descent. Ties keep the earlier value, so two runs agree byte for
// byte. Any failure falls back to a persistence forecast like the house baseline's, so a forecast is always written.
//
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const HOUR = 3_600_000;
const FLOOR = 1e-4;
const TOP = 10;
const MAX_ROWS = 100;
/** The most recent origins the backtest scores. */
const MAX_ORIGINS = 72;
/** Origins this many hours older than the last read weigh 1/e as much. */
const ORIGIN_DECAY_H = 72;

interface Read {
  t: number;
  ids: number[];
  pos: Map<number, number>;
}

interface Params {
  c: number;
  s0: number;
  s1: number;
  eps: number;
  nu: number;
  alpha: number;
  w: number;
}

interface Row {
  universe_id: number;
  place: number;
  p: number;
}

const GRID: { [K in keyof Params]: number[] } = {
  c: [0.5, 2, 5, 10, 30, 100],
  s0: [0.2, 0.35, 0.5, 0.8, 1.2, 1.8, 2.5, 3.5],
  s1: [0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2],
  eps: [0, 0.01, 0.03, 0.06, 0.1, 0.15, 0.25],
  nu: [0.005, 0.01, 0.02, 0.04, 0.08, 0.12],
  alpha: [0, 0.15, 0.3, 0.5, 0.75, 1],
  w: [6, 12, 24, 48],
};

const DEFAULTS: Params = { c: 5, s0: 0.8, s1: 0.2, eps: 0.03, nu: 0.02, alpha: 0.3, w: 12 };

function loadReads(text: string): Read[] {
  const reads: Read[] = [];
  for (const line of text.split('\n')) {
    const s = line.trim();
    if (s === '') continue;
    let o: any;
    try {
      o = JSON.parse(s);
    } catch {
      continue;
    }
    if (o === null || typeof o !== 'object' || o.ok === false || !Array.isArray(o.top_trending)) continue;
    const t = Date.parse(o.at);
    if (!Number.isFinite(t)) continue;
    const rows: Array<{ id: number; rank: number; i: number }> = [];
    (o.top_trending as any[]).forEach((r: any, i: number) => {
      if (r === null || typeof r !== 'object') return;
      const id = Number(r.universe_id);
      if (!Number.isSafeInteger(id) || id <= 0) return;
      const rank = Number(r.rank);
      rows.push({ id, rank: Number.isFinite(rank) ? rank : i + 1, i });
    });
    rows.sort((a, b) => a.rank - b.rank || a.i - b.i);
    const ids: number[] = [];
    const pos = new Map<number, number>();
    for (const r of rows) {
      if (pos.has(r.id)) continue;
      ids.push(r.id);
      pos.set(r.id, ids.length);
    }
    if (ids.length === 0) continue;
    reads.push({ t, ids, pos });
  }
  // Array.prototype.sort is stable, so reads with the same time keep the file's order.
  reads.sort((a, b) => a.t - b.t);
  return reads;
}

function warp(r: number, c: number): number {
  return (10 + c) * Math.log(r + c);
}

/** The index of the read nearest w hours before reads[o] and at least w/2 hours before it, or -1. */
function lookback(reads: Read[], o: number, w: number): number {
  const target = reads[o]!.t - w * HOUR;
  const latest = reads[o]!.t - (w / 2) * HOUR;
  let best = -1;
  let bestGap = Infinity;
  for (let j = o - 1; j >= 0; j--) {
    const t = reads[j]!.t;
    if (t > latest) continue;
    const gap = Math.abs(t - target);
    if (gap < bestGap) {
      bestGap = gap;
      best = j;
    }
    if (t < target) break;
  }
  return best;
}

/**
 * The forecast from reads[o] for the given hour offsets: P[k - 1][g] for place k and the g-th game of named (the first
 * named.length games of reads[o]'s chart, in order). Each place sums to 1 - nu.
 */
function forecast(reads: Read[], o: number, back: number, offsets: number[], p: Params, places: number, n: number): Float64Array[] {
  const cur = reads[o]!;
  const prev = back >= 0 ? reads[back]! : null;
  const gap = prev ? (cur.t - prev.t) / HOUR : 0;
  const y0 = new Float64Array(n);
  const v = new Float64Array(n);
  for (let g = 0; g < n; g++) {
    y0[g] = warp(g + 1, p.c);
    if (prev !== null && gap > 0) {
      const rp = prev.pos.get(cur.ids[g]!) ?? prev.ids.length + 1;
      v[g] = (y0[g]! - warp(rp, p.c)) / gap;
    }
  }
  const fk = new Float64Array(places);
  for (let k = 0; k < places; k++) fk[k] = warp(k + 1, p.c);
  const top = warp(1, p.c);
  const maxShift = (10 + p.c) * Math.log(3);
  const P: Float64Array[] = [];
  for (let k = 0; k < places; k++) P.push(new Float64Array(n));
  const y = new Float64Array(n);
  const d = new Float64Array(n);
  const flat = p.eps / n;
  for (const h of offsets) {
    const s = p.s0 + p.s1 * Math.sqrt(h);
    for (let g = 0; g < n; g++) {
      let shift = p.alpha * v[g]! * h;
      if (shift > maxShift) shift = maxShift;
      else if (shift < -maxShift) shift = -maxShift;
      const yg = y0[g]! + shift;
      y[g] = yg < top ? top : yg;
    }
    for (let k = 0; k < places; k++) {
      let dmin = Infinity;
      for (let g = 0; g < n; g++) {
        const dist = Math.abs(fk[k]! - y[g]!);
        d[g] = dist;
        if (dist < dmin) dmin = dist;
      }
      let sum = 0;
      for (let g = 0; g < n; g++) {
        const e = Math.exp(-(d[g]! - dmin) / s);
        d[g] = e;
        sum += e;
      }
      const row = P[k]!;
      for (let g = 0; g < n; g++) row[g] = row[g]! + (1 - p.eps) * (d[g]! / sum) + flat;
    }
  }
  const scale = (1 - p.nu) / offsets.length;
  for (let k = 0; k < places; k++) {
    const row = P[k]!;
    for (let g = 0; g < n; g++) row[g] = row[g]! * scale;
  }
  return P;
}

interface Origin {
  o: number;
  weight: number;
  evals: number[];
  back: Map<number, number>;
}

/** The weighted mean, over the origins, of the mean log score of the reads that followed each. */
function backtest(reads: Read[], origins: Origin[], offsets: number[], p: Params, places: number): number {
  let total = 0;
  let weights = 0;
  for (const og of origins) {
    const cur = reads[og.o]!;
    const n = Math.min(cur.ids.length, MAX_ROWS);
    const P = forecast(reads, og.o, og.back.get(p.w) ?? -1, offsets, p, places, n);
    let sum = 0;
    for (const e of og.evals) {
      const er = reads[e]!;
      let unnamed = 0;
      for (const id of er.ids) {
        const g = cur.pos.get(id);
        if (g === undefined || g > n) unnamed++;
      }
      for (let k = 0; k < places && k < er.ids.length; k++) {
        const g = cur.pos.get(er.ids[k]!);
        const prob = g !== undefined && g <= n ? P[k]![g - 1]! : p.nu / Math.max(1, unnamed);
        sum += Math.log(Math.max(prob, FLOOR));
      }
    }
    total += og.weight * (sum / og.evals.length);
    weights += og.weight;
  }
  return weights > 0 ? total / weights : -Infinity;
}

function fit(reads: Read[], offsets: number[], places: number): Params {
  const last = reads[reads.length - 1]!;
  const lo = Math.max(0.1, Math.min(...offsets) - 0.5);
  const hi = Math.max(...offsets) + 0.5;
  const origins: Origin[] = [];
  for (let o = reads.length - 2; o >= 0 && origins.length < MAX_ORIGINS; o--) {
    const t = reads[o]!.t;
    const evals: number[] = [];
    for (let e = o + 1; e < reads.length; e++) {
      const dh = (reads[e]!.t - t) / HOUR;
      if (dh > hi) break;
      if (dh >= lo) evals.push(e);
    }
    if (evals.length === 0) continue;
    const back = new Map<number, number>();
    for (const w of GRID.w) back.set(w, lookback(reads, o, w));
    // A recent origin whose horizon runs past the last read is scored on its early hours only: it weighs that much less.
    const coverage = Math.min(1, (last.t - t) / HOUR / hi);
    origins.push({ o, weight: coverage * Math.exp(-(last.t - t) / HOUR / ORIGIN_DECAY_H), evals, back });
  }
  if (origins.length < 3) return { ...DEFAULTS };
  let best: Params = { ...DEFAULTS };
  let bestScore = backtest(reads, origins, offsets, best, places);
  const keys = Object.keys(GRID) as Array<keyof Params>;
  for (let pass = 0; pass < 3; pass++) {
    let changed = false;
    for (const key of keys) {
      for (const value of GRID[key]) {
        if (value === best[key]) continue;
        const trial: Params = { ...best, [key]: value };
        const score = backtest(reads, origins, offsets, trial, places);
        if (score > bestScore + 1e-9) {
          bestScore = score;
          best = trial;
          changed = true;
        }
      }
    }
    if (!changed) break;
  }
  console.error(`fit: ${origins.length} origins, backtest log score ${bestScore.toFixed(4)}, params ${JSON.stringify(best)}`);
  return best;
}

/** Greedy top 10: place by place, the named game most likely there that is not chosen yet. */
function pickTop(P: Float64Array[], ids: number[], reads: Read[]): number[] {
  const chosen: number[] = [];
  const used = new Set<number>();
  for (let k = 0; k < P.length && chosen.length < TOP; k++) {
    let bg = -1;
    let bp = -1;
    const row = P[k]!;
    for (let g = 0; g < row.length; g++) {
      if (used.has(ids[g]!)) continue;
      if (row[g]! > bp) {
        bp = row[g]!;
        bg = g;
      }
    }
    if (bg >= 0) {
      chosen.push(ids[bg]!);
      used.add(ids[bg]!);
    }
  }
  // Fill from the last chart in order, then from older reads, should the places run short.
  for (let r = reads.length - 1; r >= 0 && chosen.length < TOP; r--) {
    for (const id of reads[r]!.ids) {
      if (chosen.length >= TOP) break;
      if (used.has(id)) continue;
      chosen.push(id);
      used.add(id);
    }
  }
  return chosen;
}

function rowsFrom(P: Float64Array[], ids: number[]): Row[] {
  const rows: Row[] = [];
  for (let k = 0; k < P.length; k++) {
    const row = P[k]!;
    for (let g = 0; g < row.length && g < MAX_ROWS; g++) {
      // Rounded down, so each place's sum stays at or below 1 - nu.
      const p = Math.floor(row[g]! * 1e9) / 1e9;
      if (p > 0 && p <= 1) rows.push({ universe_id: ids[g]!, place: k + 1, p });
    }
  }
  return rows;
}

/** The house baseline's persistence forecast, the fallback. */
function persistence(ids: number[], places: number): Row[] {
  const rows: Row[] = [];
  for (let place = 1; place <= places && place <= ids.length; place++) {
    rows.push({ universe_id: ids[place - 1]!, place, p: 0.5 });
    if (place === 1) {
      if (ids[1] !== undefined) rows.push({ universe_id: ids[1], place, p: 0.3 });
      continue;
    }
    rows.push({ universe_id: ids[place - 2]!, place, p: 0.15 });
    if (ids[place] !== undefined) rows.push({ universe_id: ids[place], place, p: 0.15 });
  }
  return rows;
}

function valid(rows: Row[], top10: number[], places: number): boolean {
  if (top10.length !== TOP || new Set(top10).size !== TOP) return false;
  const sums = new Array<number>(places + 1).fill(0);
  const counts = new Array<number>(places + 1).fill(0);
  const seen = new Set<string>();
  for (const r of rows) {
    if (!Number.isSafeInteger(r.universe_id) || !Number.isInteger(r.place) || r.place < 1 || r.place > places) return false;
    if (!Number.isFinite(r.p) || r.p < 0 || r.p > 1) return false;
    const key = `${r.universe_id}:${r.place}`;
    if (seen.has(key)) return false;
    seen.add(key);
    sums[r.place] += r.p;
    counts[r.place]++;
  }
  for (let k = 1; k <= places; k++) if (sums[k]! > 1 + 1e-12 || counts[k]! > MAX_ROWS) return false;
  return true;
}

function main(): void {
  const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as any;
  const places = Number.isInteger(task?.places) && task.places > 0 ? Math.min(task.places, TOP) : TOP;
  const output = typeof task?.output === 'string' && task.output !== '' ? task.output : 'forecast.json';
  const reads = loadReads(readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8'));
  if (reads.length === 0) throw new Error('no reads in the history');
  const lastIndex = reads.length - 1;
  const last = reads[lastIndex]!;

  let rows: Row[];
  let top10: number[];
  try {
    const win = task?.window ?? {};
    let from = Date.parse(win.from ?? task?.history_until);
    if (!Number.isFinite(from)) from = last.t + HOUR;
    const slotMin = Number.isFinite(Number(win.slot_minutes)) && Number(win.slot_minutes) > 0 ? Number(win.slot_minutes) : 60;
    const slots = Number.isInteger(Number(win.slots)) && Number(win.slots) > 0 ? Math.min(Number(win.slots), 240) : 24;
    // Each slot's read starts on the slot's first minute: the offset of each from the last read, in hours.
    const offsets: number[] = [];
    for (let j = 0; j < slots; j++) offsets.push(Math.max(0.05, (from + j * slotMin * 60_000 + 60_000 - last.t) / HOUR));

    const params = fit(reads, offsets, places);
    const n = Math.min(last.ids.length, MAX_ROWS);
    const P = forecast(reads, lastIndex, lookback(reads, lastIndex, params.w), offsets, params, places, n);
    const ids = last.ids.slice(0, n);
    rows = rowsFrom(P, ids);
    top10 = pickTop(P, ids, reads);
    if (!valid(rows, top10, places)) throw new Error('the model forecast failed validation');
  } catch (err) {
    console.error(`model failed, writing persistence: ${err instanceof Error ? err.message : String(err)}`);
    rows = persistence(last.ids, places);
    top10 = pickTop([], [], reads);
  }

  try {
    mkdirSync(OUTPUT_DIR, { recursive: true });
  } catch {
    // The directory is there already, or the write below says why not.
  }
  writeFileSync(join(OUTPUT_DIR, output), `${JSON.stringify({ probabilities: rows, top10 })}\n`);
}

main();
