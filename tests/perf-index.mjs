// Verifies Hive's index cache/search routines without loading embedding models.
// Usage: node tests/perf-index.mjs [path/to/db.bin]
import fs from 'fs';
import path from 'path';
import Hive from '../hive.js';

const bin = process.argv[2] || path.join(process.cwd(), 'db', 'pubs', 'pubs.bin');

if (fs.existsSync(bin)) {
  let t = performance.now();
  const index = await Hive.buildIndex(bin, 'pubs');
  console.log(`real cold build : ${(performance.now() - t).toFixed(1)} ms | entries=${index.entries.length} count=${index.count} dims=${JSON.stringify(index.dims)} bytes~${(index.bytes / 1e6).toFixed(1)}MB`);

  Hive.clearIndexCache();
  t = performance.now();
  const a = await Hive.getIndex(bin, 'pubs');
  const cold = performance.now() - t;
  t = performance.now();
  const b = await Hive.getIndex(bin, 'pubs');
  const hot = performance.now() - t;
  console.log(`real cache get  : cold=${cold.toFixed(1)} ms hot=${hot.toFixed(3)} ms same_object=${a === b} cache=${JSON.stringify(Hive.indexCacheStats())}`);
} else {
  console.log('(no real database found, skipping)');
}

// Synthetic scale benchmark: N x dim random Float32 vectors.
function makeIndex(n, dim) {
  const entries = new Array(n);
  const magnitude = Math.sqrt(dim / 3);
  for (let i = 0; i < n; i++) {
    const v = new Float32Array(dim);
    for (let d = 0; d < dim; d++) v[d] = Math.random() * 2 - 1;
    entries[i] = { id: 'e' + i, collection: 'bench', meta: {}, vector: v, magnitude };
  }
  return { entries, collections: new Map(), count: n, bytes: n * dim * 4, dims: [dim] };
}

const DIM = 768;
const N = Number(process.env.BENCH_N || 50000);
const bench = makeIndex(N, DIM);
const q = new Float32Array(DIM);
for (let d = 0; d < DIM; d++) q[d] = Math.random() * 2 - 1;

Hive.searchVector(bench, q, 25); // warm up
const runs = 5;
const times = [];
for (let r = 0; r < runs; r++) {
  const t = performance.now();
  Hive.searchVector(bench, q, 25);
  times.push(performance.now() - t);
}
const best = Math.min(...times);
console.log(`synthetic scan  : ${best.toFixed(1)} ms best-of-${runs} over ${N} x ${DIM} vectors`);
console.log(`projection      : ~${(best / N * 1000).toFixed(2)} us/vector -> ~${(best / N * 100000).toFixed(0)} ms at 100k, ~${(best / N * 1000000).toFixed(0)} ms at 1M`);
