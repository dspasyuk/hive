// Validates Hive.addTextChunks batching/slicing without loading a real model.
import Hive from '../hive.js';

const DIM = 4;
Hive.pipelines.text = async (input) => {
  const texts = Array.isArray(input) ? input : [input];
  const data = new Float32Array(texts.length * DIM);
  for (let i = 0; i < data.length; i++) data[i] = i + 1;
  return { data, dims: [texts.length, DIM] };
};
Hive.dbName = 'T';
Hive.createCollection('T');

await Hive.addTextChunks(['a', 'b', 'c', 'd', 'e'], '/tmp/x.txt');
const col = Hive.collections.get('T');

const checks = [
  ['entry count', col.length, 5],
  ['vector type', col[0].vector.constructor.name, 'Float32Array'],
  ['vector dim', col[0].vector.length, DIM],
  ['first vector', JSON.stringify(Array.from(col[0].vector)), JSON.stringify([1, 2, 3, 4])],
  ['second vector', JSON.stringify(Array.from(col[1].vector)), JSON.stringify([5, 6, 7, 8])],
  ['magnitude set', col[0].magnitude > 0, true],
  ['meta filePath', col[0].meta.filePath, '/tmp/x.txt'],
];

let ok = true;
for (const [name, got, want] of checks) {
  const pass = got === want;
  ok = ok && pass;
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}: ${got}`);
}
process.exit(ok ? 0 : 1);
