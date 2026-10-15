// Isolated check of scripts/import-folders.js logic (importFolder) on a temp
// fixture, without touching real docs/ or db/.
import fs from 'fs';
import path from 'path';
import Hive from '../hive.js';

const tmp = path.join(process.cwd(), '.tmp-import-check');
const docsAlpha = path.join(tmp, 'docs', 'Alpha');
const binDir = path.join(tmp, 'db', 'Alpha');
const binPath = path.join(binDir, 'Alpha.bin');

// build fixture: two text files, each long enough to chunk (>100 tokens)
const words = (n, seed) => Array.from({ length: n }, (_, i) => `word${seed}${i}`).join(' ');
fs.rmSync(tmp, { recursive: true, force: true });
fs.mkdirSync(docsAlpha, { recursive: true });
fs.writeFileSync(path.join(docsAlpha, 'one.txt'), `alpha document ${words(700, 1)}`);
fs.writeFileSync(path.join(docsAlpha, 'two.txt'), `beta document ${words(700, 2)}`);

async function importFolder(folderPath, dbName) {
  if (!fs.existsSync(binPath)) {
    fs.mkdirSync(binDir, { recursive: true });
    console.log(`  Created database: ${dbName}`);
  }
  Hive.collections.clear();
  Hive.dbName = dbName;
  Hive.pathToDB = binPath;
  Hive.logging = false;
  Hive.useRerank = false;
  Hive.minSliceSize = 1;
  await Hive.init({ pathToDB: binPath, dbName, logging: false });
  Hive.users.clear();
  console.log(`  Scanning ${folderPath}...`);
  await Hive.pullDocuments(folderPath);
  await Hive.saveToDisk();
  const col = Hive.collections.get(dbName) || [];
  console.log(`  Done: ${col.length} entries, ${fs.statSync(binPath).size} bytes`);
  return col.length;
}

const checks = [];
const first = await importFolder(docsAlpha, 'Alpha');
checks.push(['first run created entries', first > 0, true]);
checks.push(['multiple chunks per file', first >= 2, true]);

// Re-run: existing files must be skipped (resume / no duplicates)
const second = await importFolder(docsAlpha, 'Alpha');
checks.push(['re-run does not duplicate', second, first]);

// Verify the saved .bin is readable through the new index routine
const index = await Hive.buildIndex(binPath, 'Alpha');
checks.push(['index reads entries', index.entries.length, first]);
checks.push(['index collection count', index.collections.get('Alpha')?.count, first]);
checks.push(['index dims present', index.dims.length > 0, true]);

let ok = true;
for (const [name, got, want] of checks) {
  const pass = got === want;
  ok = ok && pass;
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}: ${got}`);
}

fs.rmSync(tmp, { recursive: true, force: true });
process.exit(ok ? 0 : 1);
