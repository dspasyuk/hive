import fs from 'fs/promises';
import path from 'path';
import doc2txt from '../doc2txt.js';

const docsDir = 'docs';
const outDir = 'docs_txt_output';

await fs.mkdir(outDir, { recursive: true });

async function processDir(dir, baseRel) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    const relPath = baseRel ? baseRel + '/' + entry.name : entry.name;
    if (entry.isDirectory()) {
      await processDir(fullPath, relPath);
    } else {
      const result = await doc2txt.extractTextFromFile(fullPath);
      const outName = relPath.replace(/[/\\]/g, '_') + '.txt';
      const outPath = path.join(outDir, outName);
      await fs.writeFile(outPath, result.text || '', 'utf8');
      console.log('Wrote: ' + outPath + ' (' + (result.text?.length || 0) + ' chars)');
    }
  }
}

await processDir(docsDir, '');
console.log('\nDone. Output in ' + outDir + '/');
