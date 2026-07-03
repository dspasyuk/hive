#!/usr/bin/env node
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import Hive from '../hive.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const DB_DIR = path.join(ROOT, 'db');
const DOCS_DIR = path.join(ROOT, 'docs');
const IMG_DIR = path.join(ROOT, 'img');
const USERS_PATH = path.join(DB_DIR, 'hive_users.json');

async function importFolder(folderPath, dbName) {
  const binDir = path.join(DB_DIR, dbName);
  const binPath = path.join(binDir, `${dbName}.bin`);

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
}

function grantAdminAccess(dbName) {
  if (!fs.existsSync(USERS_PATH)) return;
  try {
    const data = JSON.parse(fs.readFileSync(USERS_PATH, 'utf-8'));
    let changed = false;
    for (const u of Object.values(data)) {
      if (!u.roles?.includes('root') && !u.roles?.includes('dbOwner')) continue;
      u.databases = (u.databases || []).filter(d => (typeof d === 'string' ? d : d.name) !== dbName);
      u.databases.push({ name: dbName, access: 'readWrite' });
      changed = true;
    }
    if (changed) fs.writeFileSync(USERS_PATH, JSON.stringify(data, null, 2), 'utf-8');
  } catch (e) {
    console.error('  Error updating users:', e.message);
  }
}

async function main() {
  console.log('Hive Import Folders\n');

  const sourceDirs = [];
  for (const dir of [DOCS_DIR, IMG_DIR]) {
    if (!fs.existsSync(dir)) continue;
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.isDirectory()) sourceDirs.push({ folder: path.join(dir, entry.name), name: entry.name });
    }
  }

  const topFiles = !fs.existsSync(DOCS_DIR) ? [] : fs.readdirSync(DOCS_DIR).filter(f =>
    fs.statSync(path.join(DOCS_DIR, f)).isFile() &&
    ['.txt','.pdf','.doc','.docx'].includes(path.extname(f).toLowerCase())
  );

  if (sourceDirs.length === 0 && topFiles.length === 0) {
    console.log('No document folders or files found.\n');
    console.log('Place documents in subdirectories:');
    console.log('  docs/MeetingNotes/meeting1.txt');
    console.log('  img/Screenshots/photo1.jpg');
    return;
  }

  for (const { folder, name } of sourceDirs) {
    console.log(`\n=== ${name} ===`);
    await importFolder(folder, name);
    grantAdminAccess(name);
  }

  if (topFiles.length > 0) {
    const mainBin = path.join(DB_DIR, 'Main', 'Main.bin');
    if (!fs.existsSync(mainBin) || sourceDirs.length === 0) {
      console.log(`\n=== Main ===`);
      await importFolder(DOCS_DIR, 'Main');
      grantAdminAccess('Main');
    }
  }

  console.log('\nDone!');
  process.exit(0);
}

main().catch(e => { console.error('Fatal:', e.message); process.exit(1); });
