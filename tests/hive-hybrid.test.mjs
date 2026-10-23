// Validates BM25, RRF fusion, hybrid retrieval, and batched reranking
// without loading any real models.
import assert from 'node:assert/strict';
import Hive from '../hive.js';

const vec = (...values) => Float32Array.from(values);

function entry(id, title, content, vector) {
  return {
    id,
    collection: 'Hybrid',
    vector,
    magnitude: Hive.normalize(vector),
    meta: { title, content, type: 'text' },
  };
}

const entries = [
  entry('a', 'Quarterly Report 2025', 'financial results and revenue growth', vec(1, 0, 0, 0)),
  entry('b', 'Safety Manual', 'handling chemicals safely in the laboratory', vec(0, 1, 0, 0)),
  entry('c', 'Budget Notes', 'quarterly budget forecast and planning', vec(0.9, 0.1, 0, 0)),
];
const index = { entries, collections: new Map(), count: entries.length, bytes: 0, dims: [4] };

assert.deepEqual(Hive.tokenize('Hello, WORLD_2025!'), ['hello', 'world_2025']);

const kw = Hive.searchKeyword(index, 'quarterly report', 3);
assert.equal(kw[0].document.id, 'a', 'title match should rank first');
assert.ok(kw[0].keywordScore > 0);
assert.ok(kw.every((r) => r.document && typeof r.keywordScore === 'number'));

const cached = Hive.getKeywordIndex(index);
assert.equal(Hive.getKeywordIndex(index), cached, 'keyword index should be cached');
assert.equal(Hive.searchKeyword(index, 'nonexistenttermxyz').length, 0);

const fused = Hive.rrfFuse(
  [
    [{ document: entries[0], similarity: 0.9 }, { document: entries[1], similarity: 0.8 }],
    [{ document: entries[1], keywordScore: 5 }, { document: entries[2], keywordScore: 3 }],
  ],
  { labels: ['semantic', 'keyword'] }
);
assert.equal(fused[0].document.id, 'b', 'doc appearing in both lists should win RRF');
assert.equal(fused[0].similarity, 0.8);
assert.equal(fused[0].keywordScore, 5);
assert.ok(fused[0].rrfScore > fused[1].rrfScore);
assert.deepEqual(fused[0].sources.sort(), ['keyword', 'semantic']);
assert.equal(fused.length, 3, 'results should be deduplicated by id');

const ranked = Hive.normalizeRrfScores(Hive.rrfFuse([
  [{ document: entries[0], similarity: 0.9 }, { document: entries[1], similarity: 0.8 }],
  [{ document: entries[1], keywordScore: 5 }, { document: entries[2], keywordScore: 3 }],
]));
assert.equal(ranked[0].score, 1, 'top fused result should normalize to 1');
assert.ok(ranked.every((r, i) => i === 0 || r.score <= ranked[i - 1].score), 'normalized scores should follow rank order');

const hybrid = Hive.searchHybrid(index, vec(1, 0, 0, 0), 'safety manual', 3);
assert.equal(hybrid[0].document.id, 'b', 'keyword-only match should surface in hybrid search');
assert.ok(hybrid[0].keywordScore > 0);
assert.equal(typeof hybrid[0].similarity, 'number');

const titleOnly = Hive.searchHybrid(index, vec(1, 0, 0, 0), 'budget', 3);
assert.equal(titleOnly[0].document.id, 'c', 'exact title token should be found via BM25');

const seenPairs = [];
Hive.pipelines.reranker = {
  tokenizer: (queries, options) => {
    assert.ok(Array.isArray(queries));
    assert.ok(Array.isArray(options.text_pair));
    assert.equal(queries.length, options.text_pair.length);
    seenPairs.push(...options.text_pair);
    return { pairs: options.text_pair };
  },
  model: async (inputs) => {
    const n = inputs.pairs.length;
    const data = Float32Array.from(inputs.pairs.map((p) => Number(p.slice(1))));
    return { logits: { data, dims: [n, 1] } };
  },
};

const scores = await Hive.rerank('query', ['d0', 'd1', 'd2', 'd3', 'd4'], { batchSize: 2 });
assert.deepEqual(scores, [0, 1, 2, 3, 4], 'rerank scores should align with input order');
assert.deepEqual(seenPairs, ['d0', 'd1', 'd2', 'd3', 'd4']);

Hive.users.clear();
Hive.currentUser = null;
Hive.dbName = 'Hybrid';
Hive.collections.set('Hybrid', entries);
Hive.useRerank = false;
Hive.pipelines.text = async (input) => ({ data: vec(1, 0, 0, 0) });

const results = await Hive.find('safety manual', 3);
assert.equal(results[0].document.id, 'b', 'find() should use hybrid retrieval');

console.log('PASS  tokenization');
console.log('PASS  BM25 title ranking');
console.log('PASS  keyword index caching');
console.log('PASS  RRF fusion/dedup');
console.log('PASS  RRF score normalization');
console.log('PASS  hybrid retrieval');
console.log('PASS  batched rerank pairing/order');
console.log('PASS  find() hybrid integration');
