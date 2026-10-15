import { UnpackrStream } from 'msgpackr';
import fs from 'fs';
import path from 'path';

const items = [];
const stream = fs.createReadStream('db/pubs/pubs.bin');
const unpacker = new UnpackrStream({ structuredClone: true });
stream.pipe(unpacker);

unpacker.on('data', (item) => { items.push(item); });

await new Promise((resolve, reject) => {
  unpacker.on('end', resolve);
  unpacker.on('error', reject);
});

const outDir = 'docs_txt_output';
await fs.promises.mkdir(outDir, { recursive: true });

// Save all content as one file
const allContent = items.map((item, i) => {
  const title = item.meta?.title || '';
  const filePath = item.meta?.filePath || '';
  const content = item.meta?.content || '';
  return `=== Item ${i + 1}${title ? ' - ' + title : ''} ===\nFile: ${filePath}\n\n${content}\n`;
}).join('\n\n');

const allPath = path.join(outDir, 'pubs_db_all.txt');
await fs.promises.writeFile(allPath, allContent, 'utf8');
console.log('Wrote: ' + allPath + ' (' + allContent.length + ' chars, ' + items.length + ' items)');

// Also save individual files per original filename
const byFile = {};
for (const item of items) {
  const fp = item.meta?.filePath || 'unknown';
  const name = path.basename(fp);
  if (!byFile[name]) byFile[name] = [];
  byFile[name].push(item);
}

for (const [name, fileItems] of Object.entries(byFile)) {
  const combined = fileItems.map((item, i) => {
    const title = item.meta?.title || '';
    const content = item.meta?.content || '';
    return `=== Chunk ${i + 1}: ${title} ===\n${content}`;
  }).join('\n\n');
  const outName = name.replace(/[/\\]/g, '_') + '.db.txt';
  const outPath = path.join(outDir, outName);
  await fs.promises.writeFile(outPath, combined, 'utf8');
  console.log('Wrote: ' + outPath + ' (' + combined.length + ' chars, ' + fileItems.length + ' chunks)');
}
