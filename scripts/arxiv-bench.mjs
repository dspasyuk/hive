#!/usr/bin/env node
// Downloads open-access arXiv papers and imports them into a Hive database
// for performance testing. Respects arXiv's API and download rate limits.
//
// Examples:
//   node scripts/arxiv-bench.mjs --mode abstracts --count 1000
//   node scripts/arxiv-bench.mjs --mode pdfs --count 1000 --db arxiv-pdfs
//   node scripts/arxiv-bench.mjs --mode pdfs --count 500 --query "(cat:cs.CL OR cat:cs.LG)"
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { parseArgs } from 'util';
import { parseStringPromise } from 'xml2js';
import Hive from '../hive.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const DB_DIR = path.join(ROOT, 'db');
const DOCS_DIR = path.join(ROOT, 'docs');
const UA = 'hive-arxiv-bench/1.0 (performance testing; https://github.com/dspasyuk/hive)';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function log(...a) { console.log(...a); }

function parseCli() {
  const { values } = parseArgs({
    options: {
      count: { type: 'string', default: '1000' },
      mode: { type: 'string', default: 'abstracts' },
      db: { type: 'string', default: 'arxiv' },
      query: { type: 'string', default: '(cat:cs.CL OR cat:cs.LG OR cat:cs.AI OR cat:cs.CV)' },
      start: { type: 'string', default: '0' },
      'page-size': { type: 'string', default: '100' },
      'api-delay': { type: 'string', default: '3000' },
      'dl-delay': { type: 'string', default: '1500' },
      'min-slice': { type: 'string', default: '1' },
      'dry-run': { type: 'boolean', default: false },
      'no-import': { type: 'boolean', default: false },
      help: { type: 'boolean', default: false },
    },
    allowPositionals: false,
  });
  if (values.help) {
    log(`Usage: node scripts/arxiv-bench.mjs [options]

  --count N        number of papers to fetch (default 1000)
  --mode MODE      "abstracts" (fast, ~1 chunk/paper) or "pdfs" (full text, many chunks)
  --db NAME        target Hive database name (default arxiv)
  --query Q        arXiv search_query (default cs.CL/LG/AI/CV)
  --start N        arXiv start offset (default 0)
  --page-size N    arXiv results per API page (default 100, max 2000)
  --api-delay MS   delay between API calls (default 3000)
  --dl-delay MS    delay between PDF downloads (default 1500)
  --min-slice N    Hive.minSliceSize (default 1 = index every chunk)
  --dry-run        fetch metadata only, do not download or import
  --no-import      download documents but do not import into Hive`);
    process.exit(0);
  }
  return {
    count: Number(values.count),
    mode: values.mode,
    db: values.db,
    query: values.query,
    start: Number(values.start),
    pageSize: Math.min(2000, Number(values['page-size'])),
    apiDelay: Number(values['api-delay']),
    dlDelay: Number(values['dl-delay']),
    minSlice: Number(values['min-slice']),
    dryRun: values['dry-run'],
    noImport: values['no-import'],
  };
}

async function fetchMetadata(query, start, pageSize) {
  const url = `https://export.arxiv.org/api/query?search_query=${encodeURIComponent(query)}`
    + `&start=${start}&max_results=${pageSize}&sortBy=submittedDate&sortOrder=descending`;
  const res = await fetch(url, { headers: { 'User-Agent': UA }, redirect: 'follow' });
  if (!res.ok) throw new Error(`arXiv API HTTP ${res.status}`);
  const xml = await res.text();
  const feed = (await parseStringPromise(xml)).feed || {};
  const raw = feed.entry || [];
  return raw.map((e) => {
    const links = e.link || [];
    const pdf = links.find((l) => l.$ && (l.$.type === 'application/pdf' || l.$.title === 'pdf'));
    const absUrl = (e.id && e.id[0]) || '';
    const arxivId = (absUrl.split('/abs/')[1] || absUrl.split('/').pop() || '').replace(/[^A-Za-z0-9._-]/g, '_');
    return {
      id: arxivId,
      title: (e.title?.[0] || '').replace(/\s+/g, ' ').trim(),
      abstract: (e.summary?.[0] || '').replace(/\s+/g, ' ').trim(),
      authors: (e.author || []).map((a) => a.name?.[0]).filter(Boolean),
      published: e.published?.[0] || '',
      pdfUrl: pdf ? pdf.$.href : (arxivId ? `https://arxiv.org/pdf/${arxivId}` : null),
    };
  }).filter((p) => p.id);
}

async function download(url, dest, retries = 3) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const res = await fetch(url, { headers: { 'User-Agent': UA }, redirect: 'follow' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const buf = Buffer.from(await res.arrayBuffer());
      if (buf.length < 1000) throw new Error('response too small');
      const tmp = dest + '.tmp';
      fs.writeFileSync(tmp, buf);
      fs.renameSync(tmp, dest);
      return buf.length;
    } catch (e) {
      if (attempt === retries) throw e;
      await sleep(1000 * attempt);
    }
  }
}

async function importIntoHive(dbName, files, minSlice) {
  const binDir = path.join(DB_DIR, dbName);
  const binPath = path.join(binDir, `${dbName}.bin`);
  fs.mkdirSync(binDir, { recursive: true });

  Hive.collections.clear();
  Hive.dbName = dbName;
  Hive.pathToDB = binPath;
  Hive.logging = false;
  Hive.useRerank = false;
  Hive.minSliceSize = minSlice;
  await Hive.init({ pathToDB: binPath, dbName, logging: false });
  Hive.users.clear();

  let imported = 0;
  const t0 = Date.now();
  for (const fp of files) {
    if (Hive.fileExistsInDatabase(fp)) continue;
    await Hive.readFile(fp, 'text');
    imported++;
    if (imported % 25 === 0) {
      const col = Hive.collections.get(dbName) || [];
      const rate = imported / ((Date.now() - t0) / 1000);
      log(`  imported ${imported}/${files.length} files, ${col.length} chunks, ${rate.toFixed(1)} files/s`);
    }
  }
  await Hive.saveToDisk();
  const col = Hive.collections.get(dbName) || [];
  const bytes = fs.existsSync(binPath) ? fs.statSync(binPath).size : 0;
  return { files: imported, chunks: col.length, bytes };
}

async function main() {
  const opts = parseCli();
  if (!['abstracts', 'pdfs'].includes(opts.mode)) {
    throw new Error(`Unknown --mode "${opts.mode}" (use abstracts or pdfs)`);
  }

  const outDir = path.join(DOCS_DIR, opts.db);
  fs.mkdirSync(outDir, { recursive: true });
  log(`arXiv -> Hive benchmark`);
  log(`  mode=${opts.mode} target=${opts.count} db=${opts.db} query="${opts.query}"`);
  log(`  output=${path.relative(ROOT, outDir)}/`);

  // 1. Fetch metadata (paginated, rate limited).
  const papers = [];
  let start = opts.start;
  while (papers.length < opts.count) {
    const want = Math.min(opts.pageSize, opts.count - papers.length);
    log(`  fetching metadata start=${start} max=${want}...`);
    const page = await fetchMetadata(opts.query, start, want);
    if (page.length === 0) { log('  arXiv returned no more results.'); break; }
    papers.push(...page);
    start += page.length;
    if (papers.length < opts.count) await sleep(opts.apiDelay);
  }
  const selected = papers.slice(0, opts.count);
  fs.writeFileSync(path.join(outDir, '_manifest.json'), JSON.stringify(selected, null, 2));
  log(`  metadata: ${selected.length} papers`);
  if (opts.dryRun) { log('  --dry-run: stopping before download.'); return; }

  // 2. Materialize documents on disk (resumable).
  const files = [];
  let downloaded = 0;
  for (const p of selected) {
    if (opts.mode === 'abstracts') {
      const fp = path.join(outDir, `${p.id}.txt`);
      if (!fs.existsSync(fp)) {
        fs.writeFileSync(fp, `${p.title}\n\nAuthors: ${p.authors.join(', ')}\nPublished: ${p.published}\narXiv: ${p.id}\n\nAbstract\n${p.abstract}\n`);
      }
      files.push(fp);
    } else {
      const fp = path.join(outDir, `${p.id}.pdf`);
      if (!fs.existsSync(fp)) {
        try {
          await download(p.pdfUrl, fp);
          downloaded++;
          if (downloaded % 25 === 0) log(`  downloaded ${downloaded} PDFs...`);
        } catch (e) {
          console.error(`  download failed ${p.id}: ${e.message}`);
          continue;
        }
        await sleep(opts.dlDelay);
      }
      files.push(fp);
    }
  }
  log(`  documents ready: ${files.length} (new downloads: ${downloaded})`);
  if (opts.noImport) { log('  --no-import: skipping Hive import.'); return; }

  // 3. Import into Hive.
  log('  importing into Hive (embedding + chunking)...');
  const result = await importIntoHive(opts.db, files, opts.minSlice);
  log(`\nDone.`);
  log(`  files imported : ${result.files}`);
  log(`  chunks/entries : ${result.chunks}`);
  log(`  database size  : ${(result.bytes / 1e6).toFixed(1)} MB`);
  log(`  db path        : ${path.relative(ROOT, path.join(DB_DIR, opts.db, `${opts.db}.bin`))}`);
  log(`\nSearch it: select "${opts.db}" in the web UI, or POST /api/search {dbName:"${opts.db}",query:"..."}`);
}

main().catch((e) => { console.error('Fatal:', e.message); process.exit(1); });
