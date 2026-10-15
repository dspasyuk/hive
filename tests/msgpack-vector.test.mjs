import { Packr, UnpackrStream } from 'msgpackr';

const orig = Float32Array.from([0.5, -0.25, 3.14159, 1e-8]);
const packed = new Packr({ structuredClone: true }).pack({ collection: 'x', id: '1', vector: orig, meta: {}, magnitude: 1 });

async function decode(opts) {
  return new Promise((resolve, reject) => {
    const up = new UnpackrStream(opts);
    const out = [];
    up.on('data', d => out.push(d));
    up.on('end', () => resolve(out[0]));
    up.on('error', reject);
    up.end(packed);
  });
}

for (const opts of [undefined, { structuredClone: true }]) {
  const e = await decode(opts);
  const v = e.vector;
  const asArray = Array.from(v);
  console.log(`opts=${JSON.stringify(opts)} ctor=${v.constructor.name} len=${v.length} values=${JSON.stringify(asArray)} match=${JSON.stringify(asArray) === JSON.stringify(Array.from(orig))}`);
}
