import express from 'express';
import fs from 'fs';
import path from 'path';
import { createReadStream, createWriteStream } from 'fs';
import { fileURLToPath } from 'url';
import { Packr, UnpackrStream } from 'msgpackr';
import crypto from 'crypto';
import Hive from './hive.js';
import { pipeline, CLIPTextModelWithProjection, AutoTokenizer } from '@xenova/transformers';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DB_DIR = path.join(process.cwd(), 'db');
const PORT = process.env.PORT || 4173;
const app = express();

function asyncRoute(fn) {
  return (req, res, next) => fn(req, res, next).catch(next);
}

app.use(express.json({ limit: '100mb' }));

function listDatabases() {
  if (!fs.existsSync(DB_DIR)) return [];
  return fs.readdirSync(DB_DIR).filter(d => {
    const binPath = path.join(DB_DIR, d, `${d}.bin`);
    return fs.existsSync(binPath);
  });
}

function readEntries(binPath, defaultCollection) {
  return new Promise((resolve, reject) => {
    const entries = [];
    if (!fs.existsSync(binPath)) return resolve(entries);
    const readStream = createReadStream(binPath);
    const unpackr = new UnpackrStream();
    readStream.pipe(unpackr)
      .on('data', entry => {
        if (defaultCollection && !entry.collection) entry.collection = defaultCollection;
        entries.push(entry);
      })
      .on('end', () => resolve(entries))
      .on('error', reject);
  });
}

function writeEntries(binPath, entries) {
  return new Promise((resolve, reject) => {
    const packr = new Packr();
    const tmp = binPath + '.tmp';
    const writeStream = createWriteStream(tmp);
    for (const entry of entries) {
      writeStream.write(packr.encode(entry));
    }
    writeStream.end();
    writeStream.on('finish', () => {
      fs.renameSync(tmp, binPath);
      resolve();
    });
    writeStream.on('error', reject);
  });
}

function hashPassword(password, salt) {
  return crypto.createHash('sha256').update(salt + password).digest('hex');
}

app.get('/api/databases', asyncRoute(async (req, res) => {
  const dbs = listDatabases().map(name => {
    const binPath = path.join(DB_DIR, name, `${name}.bin`);
    try {
      const stat = fs.statSync(binPath);
      return { name, size: stat.size, modified: stat.mtime };
    } catch { return null; }
  }).filter(Boolean);
  res.json(dbs);
}));

app.post('/api/databases', asyncRoute(async (req, res) => {
  const { name } = req.body;
  if (!name) return res.status(400).json({ error: 'Database name required' });
  const dir = path.join(DB_DIR, name);
  const binPath = path.join(dir, name + '.bin');
  if (fs.existsSync(binPath)) return res.status(409).json({ error: 'Database already exists' });
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(binPath, Buffer.alloc(0));
  res.json({ message: 'Database created', name });
}));

app.post('/api/databases/:name/collections', asyncRoute(async (req, res) => {
  const { name } = req.params;
  const { collection } = req.body;
  if (!collection) return res.status(400).json({ error: 'Collection name required' });
  const binPath = path.join(DB_DIR, name, `${name}.bin`);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });
  const entries = await readEntries(binPath, name);
  const exists = entries.some(e => e.collection === collection);
  if (exists) return res.status(409).json({ error: 'Collection already exists' });
  const id = crypto.randomBytes(8).toString('hex');
  entries.push({ collection, id, vector: [], meta: { title: `(empty collection: ${collection})`, content: '' }, magnitude: 0 });
  await writeEntries(binPath, entries);
  res.json({ message: 'Collection created', collection });
}));

app.post('/api/databases/:name/documents', asyncRoute(async (req, res) => {
  const { name } = req.params;
  const { data, filename, collection } = req.body;
  if (!data) return res.status(400).json({ error: 'No file data' });
  const fname = filename || 'upload.txt';
  const ext = path.extname(fname).toLowerCase();
  const imageExts = ['.png', '.jpg', '.jpeg', '.gif', '.webp'];
  const storeDir = imageExts.includes(ext) ? path.join(process.cwd(), 'img') : path.join(process.cwd(), 'docs');
  if (!fs.existsSync(storeDir)) fs.mkdirSync(storeDir, { recursive: true });
  const storePath = path.join(storeDir, fname);
  let suffix = 1;
  let finalPath = storePath;
  while (fs.existsSync(finalPath)) {
    const p = path.parse(storePath);
    finalPath = path.join(p.dir, p.name + '_' + suffix + p.ext);
    suffix++;
  }
  fs.writeFileSync(finalPath, Buffer.from(data, 'base64'));
  const binPath = path.join(DB_DIR, name, `${name}.bin`);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });
  const coll = collection || name;
  try {
    const existing = await readEntries(binPath, name);
    Hive.dbName = coll;
    Hive.pathToDB = binPath;
    Hive.logging = false;
    Hive.collections.clear();
    Hive.createCollection(coll);
    const col = Hive.collections.get(coll);
    for (const e of existing) {
      col.push(e);
    }
    await Hive.addFile(finalPath, fname);
    const newEntries = col.slice(existing.length);
    if (!newEntries.length) return res.status(500).json({ error: 'addFile created no entries' });
    const all = Hive.collections.get(coll) || [];
    await writeEntries(binPath, all);
    res.json({ message: 'Document added', file: fname, entries: newEntries.length });
  } catch (e) {
    console.error('Upload error:', e);
    res.status(500).json({ error: e.message });
  } finally {
    if (Hive.saveTimeout) { clearTimeout(Hive.saveTimeout); Hive.saveTimeout = null; }
    Hive.collections.clear();
  }
}));

app.get('/api/databases/:name', asyncRoute(async (req, res) => {
  const { name } = req.params;
  const binPath = path.join(DB_DIR, name, `${name}.bin`);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });
  const entries = await readEntries(binPath, name);
  const collections = {};
  for (const e of entries) {
    if (e.collection === '__hive__') continue;
    if (!collections[e.collection]) collections[e.collection] = [];
    collections[e.collection].push(e);
  }
  const users = entries.find(e => e.collection === '__hive__');
  res.json({
    name,
    collections: Object.entries(collections).map(([name, entries]) => ({
      name,
      count: entries.length
    })),
    userCount: users ? Object.keys(users.data || {}).length : 0
  });
}));

app.get('/api/databases/:name/entries', asyncRoute(async (req, res) => {
  const { name } = req.params;
  const { collection, limit = 50, offset = 0, search, type } = req.query;
  const binPath = path.join(DB_DIR, name, `${name}.bin`);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });
  const allEntries = await readEntries(binPath, name);
  let entries = allEntries.filter(e => e.collection !== '__hive__');
  if (collection) entries = entries.filter(e => e.collection === collection);
  if (type) entries = entries.filter(e => e.meta?.type === type);
  if (search) {
    const s = search.toLowerCase();
    entries = entries.filter(e =>
      (e.meta?.title && e.meta.title.toLowerCase().includes(s)) ||
      (e.meta?.content && e.meta.content.toLowerCase().includes(s)) ||
      (e.id && e.id.includes(s))
    );
  }
  const total = entries.length;
  entries.reverse();
  const page = entries.slice(Number(offset), Number(offset) + Number(limit));
  for (const e of page) {
    if (e.vector && !Array.isArray(e.vector)) e.vector = Array.from(e.vector);
  }
  res.json({ total, offset: Number(offset), limit: Number(limit), entries: page });
}));

app.get('/api/databases/:name/search', asyncRoute(async (req, res) => {
  const { name } = req.params;
  const { q, limit = 25, type } = req.query;
  if (!q) return res.status(400).json({ error: 'Query required (q)' });
  const binPath = path.join(DB_DIR, name, `${name}.bin`);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });
  const allEntries = await readEntries(binPath, name);
  let entries = allEntries.filter(e => e.collection !== '__hive__');
  const s = q.toLowerCase();
  const keyword = entries.filter(e =>
    (e.meta?.title && e.meta.title.toLowerCase().includes(s)) ||
    (e.meta?.content && e.meta.content.toLowerCase().includes(s)) ||
    (e.id && e.id.includes(s))
  );
  let semantic = [];
  let clipImageResults = [];
  try {
    Hive.dbName = name;
    Hive.pathToDB = binPath;
    Hive.logging = false;
    Hive.collections.clear();
    Hive.createCollection(name);
    const col = Hive.collections.get(name);
    for (const e of allEntries) {
      if (e.collection === '__hive__' || !e.vector || !e.vector.length) continue;
      col.push(e);
    }
    const results = await Hive.find(q, Number(limit));
    semantic = results.map(r => ({
      collection: r.document.collection,
      id: r.document.id,
      meta: r.document.meta,
      similarity: r.similarity,
      _source: 'semantic'
    }));
    const imageEntries = col.filter(e => e.vector && e.meta?.type === 'image');
    if (imageEntries.length > 0) {
      if (!global.__clipTextModel) {
        global.__clipTextModel = await CLIPTextModelWithProjection.from_pretrained(Hive.models.image);
        global.__clipTokenizer = await AutoTokenizer.from_pretrained(Hive.models.image);
      }
      const inputs = await global.__clipTokenizer(q, { padding: true, truncation: true });
      const out = await global.__clipTextModel(inputs);
      const queryVec = Array.from(out.text_embeds.data);
      const qMag = Math.sqrt(queryVec.reduce((s, v) => s + v * v, 0));
      for (const img of imageEntries) {
        const vec = Array.from(img.vector);
        const iMag = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
        let dot = 0;
        for (let i = 0; i < queryVec.length; i++) dot += queryVec[i] * vec[i];
        const sim = dot / (qMag * iMag || 1);
        clipImageResults.push({
          collection: img.collection, id: img.id, meta: img.meta,
          similarity: sim, _source: 'clip-image'
        });
      }
      clipImageResults.sort((a, b) => b.similarity - a.similarity);
      clipImageResults = clipImageResults.slice(0, Number(limit));
    }
  } catch (e) {
    console.error('Search error:', e.message);
  } finally {
    if (Hive.saveTimeout) { clearTimeout(Hive.saveTimeout); Hive.saveTimeout = null; }
    Hive.collections.clear();
  }
  const seen = new Set();
  const merged = [];
  for (const e of [...semantic, ...clipImageResults, ...keyword]) {
    if (!seen.has(e.id)) {
      seen.add(e.id);
      merged.push({ collection: e.collection, id: e.id, meta: e.meta, similarity: e.similarity ?? null, _source: e._source || 'keyword' });
    }
  }
  const filtered = type ? merged.filter(e => e.meta?.type === type) : merged;
  res.json({ query: q, total: filtered.length, keyword: keyword.length, semantic: semantic.length, clipImages: clipImageResults.length, entries: filtered.slice(0, Number(limit)) });
}));

app.get('/api/databases/:name/users', asyncRoute(async (req, res) => {
  const { name } = req.params;
  const binPath = path.join(DB_DIR, name, `${name}.bin`);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });
  const entries = await readEntries(binPath, name);
  const hiveEntry = entries.find(e => e.collection === '__hive__');
  const users = hiveEntry ? Object.entries(hiveEntry.data || {}).map(([user, data]) => ({
    user, roles: data.roles
  })) : [];
  res.json(users);
}));

app.post('/api/databases/:name/users', asyncRoute(async (req, res) => {
  const { name } = req.params;
  const { user, password, roles } = req.body;
  if (!user || !password) return res.status(400).json({ error: 'user and password required' });
  const binPath = path.join(DB_DIR, name, `${name}.bin`);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });
  const entries = await readEntries(binPath, name);
  let hiveIdx = entries.findIndex(e => e.collection === '__hive__');
  if (hiveIdx === -1) {
    entries.push({ collection: '__hive__', _type: 'users', data: {} });
    hiveIdx = entries.length - 1;
  }
  if (entries[hiveIdx].data[user]) return res.status(409).json({ error: 'User already exists' });
  const salt = crypto.randomBytes(8).toString('hex');
  entries[hiveIdx].data[user] = {
    password: hashPassword(password, salt),
    salt, roles: roles || ['read']
  };
  await writeEntries(binPath, entries);
  res.json({ user, roles: roles || ['read'] });
}));

app.put('/api/databases/:name/users/:username', asyncRoute(async (req, res) => {
  const { name, username } = req.params;
  const { password, roles } = req.body;
  const binPath = path.join(DB_DIR, name, `${name}.bin`);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });
  const entries = await readEntries(binPath, name);
  const hiveEntry = entries.find(e => e.collection === '__hive__');
  if (!hiveEntry || !hiveEntry.data[username]) return res.status(404).json({ error: 'User not found' });
  if (password) {
    const salt = crypto.randomBytes(8).toString('hex');
    hiveEntry.data[username].password = hashPassword(password, salt);
    hiveEntry.data[username].salt = salt;
  }
  if (roles) hiveEntry.data[username].roles = roles;
  await writeEntries(binPath, entries);
  res.json({ user: username, roles: hiveEntry.data[username].roles });
}));

app.delete('/api/databases/:name/users/:username', asyncRoute(async (req, res) => {
  const { name, username } = req.params;
  const binPath = path.join(DB_DIR, name, `${name}.bin`);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });
  const entries = await readEntries(binPath, name);
  const hiveEntry = entries.find(e => e.collection === '__hive__');
  if (!hiveEntry || !hiveEntry.data[username]) return res.status(404).json({ error: 'User not found' });
  delete hiveEntry.data[username];
  await writeEntries(binPath, entries);
  res.json({ message: 'User deleted' });
}));

app.post('/api/databases/upload', asyncRoute(async (req, res) => {
  const { name, data } = req.body;
  if (!data) return res.status(400).json({ error: 'No file data' });
  let dbName = name || 'imported';
  const dir = path.join(DB_DIR, dbName);
  const binPath = path.join(dir, dbName + '.bin');
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(binPath, Buffer.from(data, 'base64'));
  res.json({ message: 'Database imported', name: dbName, size: Buffer.byteLength(data, 'base64') });
}));

app.delete('/api/databases/:name', asyncRoute(async (req, res) => {
  const { name } = req.params;
  const binPath = path.join(DB_DIR, name, `${name}.bin`);
  const dir = path.join(DB_DIR, name);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });
  fs.unlinkSync(binPath);
  try { fs.rmdirSync(dir); } catch {}
  res.json({ message: 'Database deleted' });
}));

app.delete('/api/databases/:name/collections/:collection', asyncRoute(async (req, res) => {
  const { name, collection } = req.params;
  const binPath = path.join(DB_DIR, name, `${name}.bin`);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });
  let entries = await readEntries(binPath, name);
  const before = entries.length;
  entries = entries.filter(e => e.collection !== collection);
  if (entries.length === before) return res.status(404).json({ error: 'Collection not found' });
  await writeEntries(binPath, entries);
  res.json({ message: `Collection "${collection}" dropped`, removed: before - entries.length });
}));

app.post('/api/databases/:name/entries/delete', asyncRoute(async (req, res) => {
  const { name } = req.params;
  const { ids } = req.body;
  if (!Array.isArray(ids) || !ids.length) return res.status(400).json({ error: 'ids array required' });
  const binPath = path.join(DB_DIR, name, `${name}.bin`);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });
  let entries = await readEntries(binPath, name);
  const before = entries.length;
  entries = entries.filter(e => !ids.includes(e.id));
  await writeEntries(binPath, entries);
  res.json({ message: `${before - entries.length} entries deleted`, removed: before - entries.length });
}));

app.get('/api/databases/:name/collections', asyncRoute(async (req, res) => {
  const { name } = req.params;
  const binPath = path.join(DB_DIR, name, `${name}.bin`);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });
  const entries = await readEntries(binPath, name);
  const collections = {};
  for (const e of entries) {
    if (e.collection === '__hive__') continue;
    if (!collections[e.collection]) collections[e.collection] = { count: 0, types: new Set() };
    collections[e.collection].count++;
    if (e.meta?.type) collections[e.collection].types.add(e.meta.type);
  }
  res.json(Object.entries(collections).map(([name, c]) => ({
    name, count: c.count, types: [...c.types]
  })));
}));

app.get('/api/file', (req, res) => {
  const p = req.query.path;
  if (!p) return res.status(400).json({ error: 'path required' });
  const abs = path.resolve(p);
  if (!fs.existsSync(abs)) return res.status(404).json({ error: 'File not found' });
  const ext = path.extname(abs).toLowerCase();
  const mime = { '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.gif': 'image/gif', '.webp': 'image/webp', '.svg': 'image/svg+xml', '.txt': 'text/plain', '.pdf': 'application/pdf' };
  const type = mime[ext] || 'application/octet-stream';
  if (type.startsWith('image/') || type === 'application/pdf' || type === 'text/plain') {
    if (type.startsWith('text')) res.set('Content-Type', type + '; charset=utf-8');
    else res.set('Content-Type', type);
    res.sendFile(abs);
  } else {
    res.json({ error: 'Unsupported file type' });
  }
});

app.use(express.static(path.join(__dirname, 'admin')));

app.use((err, req, res, next) => {
  console.error('API error:', err.message);
  res.status(500).json({ error: err.message || 'Internal server error' });
});

app.listen(PORT, () => {
  console.log(`Hive Admin: http://localhost:${PORT}`);
});
