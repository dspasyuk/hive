// Validates Hive.find uses the shared searchVector path (no model needed for
// vector queries with rerank disabled).
import Hive from '../hive.js';

Hive.users.clear();
Hive.useRerank = false;
Hive.pipelines.reranker = null;
Hive.dbName = 'FindTest';
Hive.collections.set('FindTest', [
  { id: 'x', vector: Float32Array.from([1, 0, 0]), magnitude: 1, meta: { title: 'x' } },
  { id: 'y', vector: Float32Array.from([0, 1, 0]), magnitude: 1, meta: { title: 'y' } },
  { id: 'z', vector: Float32Array.from([0.9, 0.1, 0]), magnitude: Math.hypot(0.9, 0.1), meta: { title: 'z' } },
]);

const results = await Hive.find(Float32Array.from([1, 0, 0]), 2);

const checks = [
  ['top id', results[0]?.document.id, 'x'],
  ['second id', results[1]?.document.id, 'z'],
  ['top similarity ~1', Math.abs(results[0].similarity - 1) < 1e-6, true],
  ['length', results.length, 2],
];

let ok = true;
for (const [name, got, want] of checks) {
  const pass = got === want;
  ok = ok && pass;
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}: ${got}`);
}
process.exit(ok ? 0 : 1);
