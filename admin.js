#!/usr/bin/env node
import express from 'express';
import fs from 'fs';
import path from 'path';
import { createReadStream, createWriteStream } from 'fs';
import { fileURLToPath } from 'url';
import { Packr, UnpackrStream } from 'msgpackr';
import crypto from 'crypto';
import Hive from './hive.js';
import { pipeline, CLIPTextModelWithProjection, AutoTokenizer } from '@xenova/transformers';
import session from 'express-session';
import FileStore from 'session-file-store';
import multer from 'multer';
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DB_DIR = path.join(process.cwd(), 'db');
const GLOBAL_USERS_PATH = path.join(DB_DIR, 'hive_users.json');
const PORT = process.env.PORT || 3030;
const app = express();

const upload = multer({ dest: 'uploads/' });
const FileStoreSession = FileStore(session);

function dbBinPath(name) {
  return path.join(DB_DIR, name, `${name}.bin`);
}

// Convert a cached index entry into a JSON-safe API document (vectors are
// Float32Array in the index, which JSON.stringify turns into an object).
function serializeDoc(doc, dbName) {
  const out = {
    id: doc.id,
    collection: doc.collection,
    meta: doc.meta,
    magnitude: doc.magnitude,
    vector: doc.vector ? Array.from(doc.vector) : [],
  };
  if (dbName) out._db = dbName;
  return out;
}

// CLIP text embedding for text->image search (model cached on the process).
async function clipTextEmbedding(text) {
  if (!global.__clipTextModel) {
    global.__clipTextModel = await CLIPTextModelWithProjection.from_pretrained(Hive.models.image);
    global.__clipTokenizer = await AutoTokenizer.from_pretrained(Hive.models.image);
  }
  const inputs = await global.__clipTokenizer(text, { padding: true, truncation: true });
  const out = await global.__clipTextModel(inputs);
  return Hive.toFloat32(Array.from(out.text_embeds.data));
}

function asyncRoute(fn) {
  return (req, res, next) => fn(req, res, next).catch(next);
}

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ extended: true }));

app.use(session({
  store: new FileStoreSession({ path: path.join(__dirname, 'sessions'), retries: 0, reapInterval: -1 }),
  secret: 'hive-session-secret-change-in-production',
  resave: false,
  saveUninitialized: false,
  rolling: true,
  cookie: { secure: false, maxAge: 24 * 60 * 60 * 1000, httpOnly: true, sameSite: 'lax' }
}));

app.use((req, res, next) => {
  if (req.session?.user?.username) {
    Hive.currentUser = req.session.user.username;
  }
  next();
});

function loggedIn(req, res, next) {
  if (req.session.loggedin) return next();
  if (req.xhr || req.path.startsWith('/api/')) {
    return res.status(401).json({ error: 'Not authenticated' });
  }
  res.redirect('/login');
}

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
    const packr = new Packr({ structuredClone: true });
    const tmp = binPath + '.tmp';
    const ws = createWriteStream(tmp);
    let streamError = null;
    ws.on('error', (err) => { streamError = err; });
    for (const entry of entries) {
      ws.write(packr.encode(entry));
    }
    ws.end();
    ws.on('finish', () => {
      if (streamError) return reject(streamError);
      fs.renameSync(tmp, binPath);
      Hive.invalidateIndex(binPath);
      resolve();
    });
  });
}

function hashPassword(password, salt) {
  return crypto.createHash('sha256').update(salt + password).digest('hex');
}

// Fast saves for admin operations
Hive.saveInterval = 100;

// Persist users to a dedicated JSON file (immediate, survives restarts)
function saveGlobalUsers() {
  if (!fs.existsSync(DB_DIR)) fs.mkdirSync(DB_DIR, { recursive: true });
  const data = {};
  for (const [username, userData] of Hive.users) {
    data[username] = {
      password: userData.password,
      salt: userData.salt,
      roles: userData.roles,
      databases: serializeDbAccess(userData.databases)
    };
  }
  fs.writeFileSync(GLOBAL_USERS_PATH, JSON.stringify(data, null, 2), 'utf-8');
}

function loadGlobalUsers() {
  if (!fs.existsSync(GLOBAL_USERS_PATH)) return;
  try {
    const data = JSON.parse(fs.readFileSync(GLOBAL_USERS_PATH, 'utf-8'));
    for (const [username, userData] of Object.entries(data)) {
      if (!Hive.users.has(username)) {
        Hive.users.set(username, {
          password: userData.password,
          salt: userData.salt,
          roles: userData.roles || ['read'],
          databases: normalizeDbAccess(userData.databases)
        });
      }
    }
  } catch (e) {
    console.error('Failed to load global users:', e.message);
  }
}

// Normalize old array format to new {name,access}[] format
function normalizeDbAccess(databases) {
  if (!databases || !Array.isArray(databases)) return [];
  return databases.map(d => {
    if (typeof d === 'string') return { name: d, access: 'read' };
    return { name: d.name, access: d.access || 'read' };
  });
}

function dbAccessForUser(username, dbName) {
  const user = Hive.users.get(username);
  if (!user) return null;
  if (user.roles?.includes('root') || user.roles?.includes('dbOwner')) return 'readWrite';
  const dbs = normalizeDbAccess(user.databases);
  const entry = dbs.find(d => d.name === dbName);
  return entry ? entry.access : null;
}

function userCanAccessDb(username, dbName) {
  return dbAccessForUser(username, dbName) !== null;
}

function userCanWriteDb(username, dbName) {
  const access = dbAccessForUser(username, dbName);
  return access === 'readWrite';
}

function getUserDatabases(username) {
  const user = Hive.users.get(username);
  if (!user) return [];
  if (user.roles?.includes('root') || user.roles?.includes('dbOwner')) return listDatabases();
  return normalizeDbAccess(user.databases)
    .map(d => d.name)
    .filter(d => listDatabases().includes(d));
}

// Convert internal databases array back to storage format
function serializeDbAccess(databases) {
  if (!databases || !Array.isArray(databases)) return [];
  return databases.map(d => {
    if (typeof d === 'string') return { name: d, access: 'read' };
    return { name: d.name, access: d.access || 'read' };
  });
}

// Temporarily grant Hive write access to a user for a single operation.
// Our API layer handles auth, but Hive's internal checkAuth still validates roles.
async function withHiveWriteAccess(username, fn) {
  if (!username || !Hive.users.has(username)) return fn();
  const userData = Hive.users.get(username);
  const originalRoles = [...userData.roles];
  if (!userData.roles.includes('readWrite') && !userData.roles.includes('root') && !userData.roles.includes('dbOwner')) {
    userData.roles.push('readWrite');
  }
  try {
    return await fn();
  } finally {
    userData.roles = originalRoles;
  }
}

function requireAdmin(req, res, next) {
  const username = req.session?.user?.username;
  if (!username) return res.status(401).json({ error: 'Not authenticated' });
  const user = Hive.users.get(username);
  if (!user || (!user.roles?.includes('root') && !user.roles?.includes('dbOwner') && !user.roles?.includes('dbAdmin'))) {
    if (req.path.startsWith('/api/')) return res.status(403).json({ error: 'Admin access required' });
    return res.redirect('/search');
  }
  next();
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
  const { name, path: docsPath } = req.body;
  if (!name) return res.status(400).json({ error: 'Database name required' });
  const dir = path.join(DB_DIR, name);
  const binPath = path.join(dir, name + '.bin');
  if (fs.existsSync(binPath)) return res.status(409).json({ error: 'Database already exists' });
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(binPath, Buffer.alloc(0));
  Hive.invalidateIndex(binPath);
  const username = req.session?.user?.username;
  if (username && Hive.users.has(username)) {
    const user = Hive.users.get(username);
    if (!user.databases) user.databases = [];
    user.databases = normalizeDbAccess(user.databases);
    if (!user.databases.find(d => d.name === name)) {
      user.databases.push({ name, access: 'readWrite' });
    }
    saveGlobalUsers();
  }
  const imported = [];
  if (docsPath && fs.existsSync(docsPath)) {
    const prevMinSlice = Hive.minSliceSize;
    Hive.minSliceSize = 1;
    try {
      Hive.dbName = name;
      Hive.pathToDB = binPath;
      Hive.logging = false;
      if (!Hive.overlap || Hive.overlap === 0) {
        Hive.overlap = Math.floor(Hive.SliceSize * 0.05);
      }
      Hive.collections.clear();
      Hive.createCollection(name);
      Hive.currentUser = username;
      const stat = fs.statSync(docsPath);
      if (stat.isDirectory()) {
        await Hive.pullDocuments(docsPath);
      } else {
        await Hive.addFile(docsPath, path.basename(docsPath));
      }
      const configEntry = {
        collection: '__hive__',
        id: '__config__',
        vector: [],
        meta: { docsPath, created: new Date().toISOString() },
        magnitude: 0
      };
      Hive.collections.get(name)?.push(configEntry);
      imported.push(docsPath);
      await Hive.saveToDisk();
    } catch (e) {
      console.error('Import error:', e.message);
    } finally {
      Hive.minSliceSize = prevMinSlice;
      if (Hive.saveTimeout) { clearTimeout(Hive.saveTimeout); Hive.saveTimeout = null; }
      Hive.collections.clear();
    }
  }
  res.json({ message: 'Database created', name, imported: imported.length });
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
  const username = req.session?.user?.username;
  if (username && !userCanWriteDb(username, name)) {
    return res.status(403).json({ error: 'Write permission denied for this database' });
  }
  const binPath = path.join(DB_DIR, name, `${name}.bin`);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });

  const { data, filename, collection } = req.body;
  if (!data) return res.status(400).json({ error: 'No file data' });
  const replace = req.body.replace !== false;
  const fname = filename || 'upload.txt';
  const ext = path.extname(fname).toLowerCase();
  const imageExts = ['.png', '.jpg', '.jpeg', '.gif', '.webp'];
  const storeDir = imageExts.includes(ext) ? path.join(process.cwd(), 'img', name) : path.join(process.cwd(), 'docs', name);
  if (!fs.existsSync(storeDir)) fs.mkdirSync(storeDir, { recursive: true });
  const storePath = path.join(storeDir, fname);
  const duplicate = fs.existsSync(storePath);
  let finalPath = storePath;
  if (duplicate && !replace) {
    let suffix = 1;
    while (fs.existsSync(finalPath)) {
      const p = path.parse(storePath);
      finalPath = path.join(p.dir, p.name + '_' + suffix + p.ext);
      suffix++;
    }
  }
  const backupPath = replace && duplicate ? `${storePath}.bak` : null;
  const coll = collection || name;
  try {
    if (backupPath) {
      if (fs.existsSync(backupPath)) fs.unlinkSync(backupPath);
      fs.renameSync(finalPath, backupPath);
    }
    fs.writeFileSync(finalPath, Buffer.from(data, 'base64'));

    const existing = await readEntries(binPath, name);
    const replaced = replace && existing.some(e => e.meta?.filePath === finalPath);
    Hive.dbName = coll;
    Hive.pathToDB = binPath;
    Hive.logging = false;
    Hive.collections.clear();
    Hive.createCollection(coll);
    const col = Hive.collections.get(coll);
    const kept = replace ? existing.filter(e => e.meta?.filePath !== finalPath) : existing;
    for (const e of kept) {
      col.push(e);
    }
    await withHiveWriteAccess(username, () => Hive.addFile(finalPath, fname, { save: false }));
    const newEntries = col.slice(kept.length);
    if (!newEntries.length) throw new Error('addFile created no entries');
    const all = Hive.collections.get(coll) || [];
    await writeEntries(binPath, all);
    if (backupPath && fs.existsSync(backupPath)) fs.unlinkSync(backupPath);
    res.json({
      message: replaced ? 'Document replaced' : 'Document added',
      file: fname,
      replaced,
      entries: newEntries.length
    });
  } catch (e) {
    console.error('Upload error:', e.message);
    if (backupPath && fs.existsSync(backupPath)) {
      try {
        if (fs.existsSync(finalPath)) fs.unlinkSync(finalPath);
        fs.renameSync(backupPath, finalPath);
      } catch (restoreErr) {
        console.error('Failed to restore previous file:', restoreErr.message);
      }
    } else {
      try { if (fs.existsSync(finalPath)) fs.unlinkSync(finalPath); } catch {}
    }
    res.status(500).json({ error: e.message });
  } finally {
    if (Hive.saveTimeout) { clearTimeout(Hive.saveTimeout); Hive.saveTimeout = null; }
    Hive.collections.clear();
  }
}));

app.get('/api/databases/:name', asyncRoute(async (req, res) => {
  const { name } = req.params;
  const binPath = dbBinPath(name);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });
  const index = await Hive.getIndex(binPath, name);
  const users = index.entries.find(e => e.collection === '__hive__');
  res.json({
    name,
    collections: [...index.collections.entries()].map(([cname, c]) => ({
      name: cname,
      count: c.count
    })),
    userCount: users && users.data ? Object.keys(users.data).length : 0
  });
}));

app.get('/api/databases/:name/entries', asyncRoute(async (req, res) => {
  const { name } = req.params;
  const { collection, limit = 50, offset = 0, search, type } = req.query;
  const binPath = dbBinPath(name);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });
  const index = await Hive.getIndex(binPath, name);
  const s = search ? search.toLowerCase() : null;
  const matched = [];
  for (let i = index.entries.length - 1; i >= 0; i--) {
    const e = index.entries[i];
    if (e.collection === '__hive__') continue;
    if (collection && e.collection !== collection) continue;
    if (type && e.meta?.type !== type) continue;
    if (s && !(
      (e.meta?.title && e.meta.title.toLowerCase().includes(s)) ||
      (e.meta?.content && e.meta.content.toLowerCase().includes(s)) ||
      (e.id && e.id.includes(s))
    )) continue;
    matched.push(e);
  }
  const start = Number(offset);
  const page = matched.slice(start, start + Number(limit)).map(e => ({
    id: e.id,
    collection: e.collection,
    meta: e.meta,
    magnitude: e.magnitude,
    vector: e.vector ? Array.from(e.vector) : [],
  }));
  res.json({ total: matched.length, offset: start, limit: Number(limit), entries: page });
}));

app.get('/api/databases/:name/search', asyncRoute(async (req, res) => {
  const { name } = req.params;
  const { q, limit = 25, type } = req.query;
  if (!q) return res.status(400).json({ error: 'Query required (q)' });
  const binPath = dbBinPath(name);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });
  const index = await Hive.getIndex(binPath, name);
  const limitK = Math.max(1, Number(limit) || 25);
  const candidateK = Math.max(limitK * 4, 50);
  const keyword = Hive.searchKeyword(index, q, candidateK).map(r => ({
    collection: r.document.collection,
    id: r.document.id,
    meta: r.document.meta,
    keywordScore: r.keywordScore
  }));
  let semantic = [];
  let clipImageResults = [];
  try {
    const queryVec = await Hive.embed(q, 'text');
    semantic = Hive.searchVector(index, queryVec, candidateK).map(r => ({
      collection: r.document.collection,
      id: r.document.id,
      meta: r.document.meta,
      similarity: r.similarity
    }));
    const imageEntries = index.entries.filter(e => e.vector && e.meta?.type === 'image');
    if (imageEntries.length > 0) {
      const clipVec = await clipTextEmbedding(q);
      clipImageResults = Hive.searchVector({ entries: imageEntries }, clipVec, candidateK).map(r => ({
        collection: r.document.collection,
        id: r.document.id,
        meta: r.document.meta,
        similarity: r.similarity
      }));
    }
  } catch (e) {
    console.error('Search error:', e.message);
  }
  const fused = Hive.rrfFuse(
    [semantic, clipImageResults, keyword],
    { labels: ['semantic', 'clip-image', 'keyword'] }
  );
  const merged = Hive.normalizeRrfScores(fused).map((r) => {
    const sources = r.sources || [];
    let source = 'semantic';
    if (sources.includes('semantic') && sources.includes('keyword')) source = 'hybrid';
    else if (sources.includes('keyword')) source = 'keyword';
    else if (sources.includes('clip-image')) source = 'clip-image';
    return {
      collection: r.document.collection,
      id: r.document.id,
      meta: r.document.meta,
      similarity: r.similarity ?? null,
      keywordScore: r.keywordScore ?? null,
      rrfScore: r.rrfScore,
      score: r.score,
      _source: source
    };
  });
  const filtered = type ? merged.filter(e => e.meta?.type === type) : merged;
  res.json({
    query: q,
    total: filtered.length,
    keyword: keyword.length,
    semantic: semantic.length,
    clipImages: clipImageResults.length,
    entries: filtered.slice(0, limitK)
  });
}));

app.get('/api/databases/:name/users', asyncRoute(async (req, res) => {
  const { name } = req.params;
  const binPath = dbBinPath(name);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });
  const index = await Hive.getIndex(binPath, name);
  const hiveEntry = index.entries.find(e => e.collection === '__hive__');
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
  Hive.invalidateIndex(binPath);
  res.json({ message: 'Database imported', name: dbName, size: Buffer.byteLength(data, 'base64') });
}));

app.delete('/api/databases/:name', asyncRoute(async (req, res) => {
  const { name } = req.params;
  const username = req.session?.user?.username;
  if (username) {
    const user = Hive.users.get(username);
    if (!user || (!user.roles?.includes('root') && !user.roles?.includes('dbOwner'))) {
      return res.status(403).json({ error: 'Admin access required to delete databases' });
    }
  }
  const binPath = path.join(DB_DIR, name, `${name}.bin`);
  const dir = path.join(DB_DIR, name);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });
  fs.unlinkSync(binPath);
  try { fs.rmdirSync(dir); } catch {}
  Hive.invalidateIndex(binPath);
  res.json({ message: 'Database deleted' });
}));

app.delete('/api/databases/:name/collections/:collection', loggedIn, asyncRoute(async (req, res) => {
  const { name, collection } = req.params;
  const username = req.session.user.username;
  if (!userCanWriteDb(username, name)) return res.status(403).json({ error: 'Access denied' });
  const binPath = path.join(DB_DIR, name, `${name}.bin`);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });
  let entries = await readEntries(binPath, name);
  const before = entries.length;
  entries = entries.filter(e => e.collection !== collection);
  if (entries.length === before) return res.status(404).json({ error: 'Collection not found' });
  await writeEntries(binPath, entries);
  res.json({ message: `Collection "${collection}" dropped`, removed: before - entries.length });
}));

app.post('/api/databases/:name/entries/delete', loggedIn, asyncRoute(async (req, res) => {
  const { name } = req.params;
  const username = req.session.user.username;
  if (!userCanWriteDb(username, name)) return res.status(403).json({ error: 'Access denied' });
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
  const binPath = dbBinPath(name);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });
  const index = await Hive.getIndex(binPath, name);
  res.json([...index.collections.entries()].map(([cname, c]) => ({
    name: cname, count: c.count, types: [...c.types]
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

// --- REBUILD DATABASE ---

const rebuildJobs = new Map();

app.post('/api/databases/:name/rebuild', loggedIn, asyncRoute(async (req, res) => {
  const { name } = req.params;
  const username = req.session.user.username;
  if (!userCanWriteDb(username, name)) return res.status(403).json({ error: 'Access denied' });

  const binPath = path.join(DB_DIR, name, `${name}.bin`);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });

  const jobId = crypto.randomBytes(8).toString('hex');
  const job = { status: 'running', total: 0, done: 0, errors: [], warnings: [] };
  rebuildJobs.set(jobId, job);

  res.json({ jobId, message: 'Rebuild started' });

  processRebuild(jobId, name, binPath, username).catch(e => {
    job.status = 'error';
    job.errors.push(e.message);
  });
}));

app.get('/api/rebuild-status/:jobId', loggedIn, asyncRoute(async (req, res) => {
  const job = rebuildJobs.get(req.params.jobId);
  if (!job) return res.status(404).json({ error: 'Job not found' });
  if (job.status === 'complete' || job.status === 'error') {
    setTimeout(() => rebuildJobs.delete(req.params.jobId), 60000);
  }
  res.json(job);
}));

async function processRebuild(jobId, dbName, binPath, username) {
  const job = rebuildJobs.get(jobId);
  if (!job) return;

  const prevMinSlice = Hive.minSliceSize;
  Hive.minSliceSize = 1;

  try {
    const entries = await readEntries(binPath, dbName);
    const configEntry = entries.find(e => e.id === '__config__');
    const docsPath = configEntry?.meta?.docsPath;

    Hive.dbName = dbName;
    Hive.pathToDB = binPath;
    Hive.logging = false;
    if (!Hive.overlap || Hive.overlap === 0) {
      Hive.overlap = Math.floor(Hive.SliceSize * 0.05);
    }
    Hive.collections.clear();
    Hive.createCollection(dbName);
    Hive.currentUser = username;

    if (docsPath && fs.existsSync(docsPath) && fs.statSync(docsPath).isDirectory()) {
      job.status = 'scanning';
      job.message = `Re-scanning ${docsPath}...`;
      await Hive.pullDocuments(docsPath);
      const col = Hive.collections.get(dbName) || [];
      col.push(configEntry);
      job.done = col.filter(e => e.meta?.filePath).length;
      job.total = job.done;
    } else {
      const filePaths = [...new Set(entries.filter(e => e.meta?.filePath).map(e => e.meta.filePath))];
      const nonFileEntries = entries.filter(e => !e.meta?.filePath);
      job.total = filePaths.length;

      for (const e of nonFileEntries) {
        if (e.id !== '__config__') Hive.collections.get(dbName).push(e);
      }

      for (let i = 0; i < filePaths.length; i++) {
        const fp = filePaths[i];
        if (!fs.existsSync(fp)) {
          job.warnings.push(`File not found: ${fp}`);
          job.done = i + 1;
          continue;
        }
        try {
          await Hive.addFile(fp, path.basename(fp));
        } catch (e) {
          job.warnings.push(`Failed: ${fp} - ${e.message}`);
        }
        job.done = i + 1;
      }
    }

    const all = Hive.collections.get(dbName) || [];
    await writeEntries(binPath, all);
    job.status = 'complete';
    if (!job.message) job.message = `Processed ${all.filter(e => e.meta?.filePath).length} files`;
  } catch (e) {
    job.status = 'error';
    job.errors.push(e.message);
  } finally {
    Hive.minSliceSize = prevMinSlice;
    if (Hive.saveTimeout) { clearTimeout(Hive.saveTimeout); Hive.saveTimeout = null; }
    Hive.collections.clear();
  }
}

// --- SEARCH UI ROUTES ---

// Login
app.get('/login', (req, res) => {
  res.render('login', { baseurl: '', title: 'Login', error: null });
});

app.post('/login', async (req, res) => {
  const { username, password } = req.body;
  if (!username || !password) {
    return res.render('login', { baseurl: '', title: 'Login', error: 'Please enter both username and password.' });
  }
  try {
    Hive.auth(username, password);
    req.session.loggedin = true;
    req.session.user = { username };
    req.session.save((err) => {
      if (err) { console.error('Session save error:', err); return res.status(500).send('Session error'); }
      if (req.xhr) return res.json({ success: true });
      res.redirect('/search');
    });
  } catch (error) {
    res.render('login', { baseurl: '', title: 'Login', error: 'Invalid username or password.' });
  }
});

app.get('/logout', (req, res) => {
  Hive.logout();
  req.session.destroy();
  res.redirect('/');
});

// Search page (protected)
app.get('/search', loggedIn, (req, res) => {
  const username = req.session.user.username;
  const user = Hive.users.get(username);
  const isAdmin = user && (user.roles?.includes('root') || user.roles?.includes('dbOwner') || user.roles?.includes('dbAdmin'));
  res.render('search', { username, isAdmin });
});

// List users with access to a database (DB owner only)
app.get('/api/search/databases/:name/users', loggedIn, asyncRoute(async (req, res) => {
  const { name } = req.params;
  const username = req.session.user.username;
  if (!userCanWriteDb(username, name)) return res.status(403).json({ error: 'Write access required to manage this database' });
  const result = [];
  for (const [u, data] of Hive.users) {
    if (u === username) continue;
    const dbs = normalizeDbAccess(data.databases || []);
    const entry = dbs.find(d => d.name === name);
    if (entry) result.push({ user: u, access: entry.access });
  }
  res.json(result);
}));

// Set user access to a database (DB owner only)
app.put('/api/search/databases/:name/users/:targetUser', loggedIn, asyncRoute(async (req, res) => {
  const { name, targetUser } = req.params;
  const { access } = req.body;
  const username = req.session.user.username;
  if (!userCanWriteDb(username, name)) return res.status(403).json({ error: 'Write access required to manage this database' });
  if (!Hive.users.has(targetUser)) return res.status(404).json({ error: 'User not found' });
  const userData = Hive.users.get(targetUser);
  userData.databases = normalizeDbAccess(userData.databases);
  if (!access || access === 'none') {
    userData.databases = userData.databases.filter(d => d.name !== name);
  } else {
    const existing = userData.databases.find(d => d.name === name);
    if (existing) existing.access = access;
    else userData.databases.push({ name, access });
  }
  saveGlobalUsers();
  res.json({ user: targetUser, database: name, access: access || 'none' });
}));

// List databases with document counts and user's access level (protected)
app.get('/api/search/databases', loggedIn, asyncRoute(async (req, res) => {
  const username = req.session.user.username;
  const dbs = getUserDatabases(username);
  const result = [];
  for (const name of dbs) {
    const binPath = dbBinPath(name);
    let docCount = 0;
    try {
      const index = await Hive.getIndex(binPath, name);
      docCount = index.count;
    } catch { docCount = 0; }
    const access = dbAccessForUser(username, name);
    result.push({ name, documentCount: docCount, access });
  }
  res.json(result);
}));

// Semantic search (protected)
app.post('/api/search', loggedIn, asyncRoute(async (req, res) => {
  const { query, dbName, searchType } = req.body;
  if (!query) return res.status(400).json({ error: 'Query required' });

  const username = req.session.user.username;
  const searchDb = dbName === '*' ? null : dbName;
  const dbs = searchDb ? (userCanAccessDb(username, searchDb) ? [searchDb] : []) : getUserDatabases(username);

  const isImageSearch = searchType === 'image';
  let queryVec = null;
  let clipVec = null;
  try {
    if (isImageSearch) clipVec = await clipTextEmbedding(query);
    else queryVec = await Hive.embed(query, 'text');
  } catch (e) {
    console.error('Query embedding error:', e.message);
    return res.status(500).json({ error: 'Failed to embed query: ' + e.message });
  }

  const semanticAll = [];
  const keywordAll = [];
  for (const name of dbs) {
    const binPath = dbBinPath(name);
    if (!fs.existsSync(binPath)) continue;
    try {
      const index = await Hive.getIndex(binPath, name);
      if (isImageSearch) {
        const imageEntries = index.entries.filter(e => e.vector && e.meta?.type === 'image');
        for (const r of Hive.searchVector({ entries: imageEntries }, clipVec, 50)) {
          semanticAll.push({ document: serializeDoc(r.document, name), similarity: r.similarity });
        }
      } else {
        for (const r of Hive.searchVector(index, queryVec, 50)) {
          semanticAll.push({ document: serializeDoc(r.document, name), similarity: r.similarity });
        }
        for (const r of Hive.searchKeyword(index, query, 50)) {
          if (r.document.meta?.type === 'image') continue;
          keywordAll.push({ document: serializeDoc(r.document, name), keywordScore: r.keywordScore });
        }
      }
    } catch (e) {
      console.error(`Search error in "${name}":`, e.message);
    }
  }

  let allResults;
  if (isImageSearch) {
    allResults = semanticAll
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, 25)
      .map((r) => ({ ...r, score: r.similarity }));
  } else {
    semanticAll.sort((a, b) => b.similarity - a.similarity);
    keywordAll.sort((a, b) => b.keywordScore - a.keywordScore);
    const fused = Hive.rrfFuse(
      [semanticAll.slice(0, 100), keywordAll.slice(0, 100)],
      { labels: ['semantic', 'keyword'] }
    );
    if (fused.length) {
      allResults = Hive.normalizeRrfScores(fused.slice(0, 25)).map((r) => ({
        document: r.document,
        similarity: r.similarity,
        keywordScore: r.keywordScore,
        rrfScore: r.rrfScore,
        score: r.score,
        sources: r.sources,
      }));
    } else {
      allResults = semanticAll.slice(0, 25).map((r) => ({ ...r, score: r.similarity }));
    }
  }
  res.json(allResults);
}));

// Image search by file upload (protected)
app.post('/api/search/image', loggedIn, upload.single('image'), asyncRoute(async (req, res) => {
  const file = req.file;
  const dbName = req.body.dbName || '*';
  const username = req.session.user.username;

  if (!file) return res.status(400).json({ error: 'No image file provided' });

  const ext = path.extname(file.originalname).toLowerCase();
  const tempPath = file.path + ext;
  fs.renameSync(file.path, tempPath);

  try {
    const vectorData = await Hive.getVector(tempPath, Hive.TransOptions);
    const queryVec = vectorData.data;

    const searchDb = dbName === '*' ? null : dbName;
    const dbs = searchDb ? (userCanAccessDb(username, searchDb) ? [searchDb] : []) : getUserDatabases(username);
    let allResults = [];

    for (const name of dbs) {
      const binPath = dbBinPath(name);
      if (!fs.existsSync(binPath)) continue;
      const index = await Hive.getIndex(binPath, name);
      const results = Hive.searchVector(index, queryVec, 25);
      for (const r of results) {
        allResults.push({ document: serializeDoc(r.document, name), similarity: r.similarity });
      }
    }

    allResults.sort((a, b) => b.similarity - a.similarity);
    allResults = allResults.slice(0, 25);
    res.json(allResults);
  } catch (e) {
    console.error('Image search error:', e);
    res.status(500).json({ error: 'Image search failed: ' + e.message });
  } finally {
    if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
  }
}));

// Add text document (protected)
app.post('/api/databases/text', loggedIn, asyncRoute(async (req, res) => {
  const { dbName, title, text } = req.body;
  if (!dbName || !title || !text) return res.status(400).json({ error: 'dbName, title, and text required' });

  const username = req.session.user.username;
  if (!userCanAccessDb(username, dbName)) return res.status(403).json({ error: 'Access denied' });
  if (!userCanWriteDb(username, dbName)) return res.status(403).json({ error: 'Write permission denied for this database' });

  const binPath = path.join(DB_DIR, dbName, `${dbName}.bin`);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });

  const docsDir = path.join(process.cwd(), 'docs', dbName);
  if (!fs.existsSync(docsDir)) fs.mkdirSync(docsDir, { recursive: true });
  const fileName = `${title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_${Date.now()}.txt`;
  const filePath = path.join(docsDir, fileName);
  fs.writeFileSync(filePath, text, 'utf8');

  try {
    const existing = await readEntries(binPath, dbName);
    Hive.dbName = dbName;
    Hive.pathToDB = binPath;
    Hive.logging = false;
    Hive.collections.clear();
    Hive.createCollection(dbName);
    const col = Hive.collections.get(dbName);
    for (const e of existing) col.push(e);
    await withHiveWriteAccess(username, () => Hive.addFile(filePath, title));
    const newEntries = col.slice(existing.length);
    if (!newEntries.length) return res.status(500).json({ error: 'addFile created no entries' });
    const all = Hive.collections.get(dbName) || [];
    await writeEntries(binPath, all);
    res.json({ message: 'Text document added', title, entries: newEntries.length });
  } catch (e) {
    console.error('Text upload error:', e.message);
    res.status(500).json({ error: e.message });
  } finally {
    if (Hive.saveTimeout) { clearTimeout(Hive.saveTimeout); Hive.saveTimeout = null; }
    Hive.collections.clear();
  }
}));

// Delete a document entry by ID (protected)
app.delete('/api/databases/:name/documents/:id', loggedIn, asyncRoute(async (req, res) => {
  const { name, id } = req.params;
  const username = req.session.user.username;
  if (!userCanWriteDb(username, name)) return res.status(403).json({ error: 'Access denied' });
  const binPath = path.join(DB_DIR, name, `${name}.bin`);
  if (!fs.existsSync(binPath)) return res.status(404).json({ error: 'Database not found' });
  let entries = await readEntries(binPath, name);
  const doc = entries.find(e => e.id === id);
  if (!doc) return res.status(404).json({ error: 'Document not found' });
  entries = entries.filter(e => e.id !== id);

  const filePath = doc.meta?.filePath || doc.meta?.href;
  if (filePath && fs.existsSync(filePath)) {
    try { fs.unlinkSync(filePath); } catch (e) { console.error('Failed to delete file:', e); }
  }

  await writeEntries(binPath, entries);
  res.json({ message: 'Document deleted', id });
}));

// --- GLOBAL USER MANAGEMENT (admin only) ---

app.get('/api/admin/users', loggedIn, requireAdmin, asyncRoute(async (req, res) => {
  const users = [];
  for (const [username, data] of Hive.users) {
    const dbs = data.databases || [];
    users.push({ user: username, roles: data.roles, databases: Array.isArray(dbs) ? normalizeDbAccess(dbs) : [] });
  }
  res.json(users);
}));

app.post('/api/admin/users', loggedIn, requireAdmin, asyncRoute(async (req, res) => {
  const { user, password, roles, databases } = req.body;
  if (!user || !password) return res.status(400).json({ error: 'user and password required' });
  if (Hive.users.has(user)) return res.status(409).json({ error: 'User already exists' });
  const salt = crypto.randomBytes(8).toString('hex');
  const normalizedDbs = Array.isArray(databases) ? databases.map(d => {
    if (typeof d === 'string') return { name: d, access: 'read' };
    return { name: d.name, access: d.access || 'read' };
  }) : [];
  Hive.users.set(user, {
    password: hashPassword(password, salt),
    salt,
    roles: roles || ['read'],
    databases: normalizedDbs
  });
  saveGlobalUsers();
  res.json({ user, roles: roles || ['read'], databases: normalizedDbs });
}));

app.delete('/api/admin/users/:username', loggedIn, requireAdmin, asyncRoute(async (req, res) => {
  const { username } = req.params;
  if (!Hive.users.has(username)) return res.status(404).json({ error: 'User not found' });
  Hive.users.delete(username);
  saveGlobalUsers();
  res.json({ message: 'User deleted' });
}));

app.put('/api/admin/users/:username/databases', loggedIn, requireAdmin, asyncRoute(async (req, res) => {
  const { username } = req.params;
  const { databases } = req.body;
  if (!Hive.users.has(username)) return res.status(404).json({ error: 'User not found' });
  const userData = Hive.users.get(username);
  const normalizedDbs = Array.isArray(databases) ? databases.map(d => {
    if (typeof d === 'string') return { name: d, access: 'read' };
    return { name: d.name, access: d.access || 'read' };
  }) : [];
  userData.databases = normalizedDbs;
  saveGlobalUsers();
  res.json({ user: username, databases: normalizedDbs });
}));

// Update user password/roles (admin only)
app.put('/api/admin/users/:username', loggedIn, requireAdmin, asyncRoute(async (req, res) => {
  const { username } = req.params;
  const { password, roles } = req.body;
  if (!Hive.users.has(username)) return res.status(404).json({ error: 'User not found' });
  const userData = Hive.users.get(username);
  if (password) {
    const salt = crypto.randomBytes(8).toString('hex');
    userData.password = hashPassword(password, salt);
    userData.salt = salt;
  }
  if (roles) userData.roles = roles;
  userData.databases = normalizeDbAccess(userData.databases);
  saveGlobalUsers();
  res.json({ user: username, roles: userData.roles, databases: userData.databases || [] });
}));

// --- END USER MANAGEMENT ---

// Init Hive and create default admin user
async function initHive() {
  // Load global users from dedicated JSON file (survives restarts)
  loadGlobalUsers();

  if (Hive.users.size === 0) {
    const salt = Hive.generateSalt();
    Hive.users.set('admin', {
      password: Hive.hashPassword('3030', salt),
      salt,
      roles: ['root'],
      databases: []
    });
    saveGlobalUsers();
    console.log('Default admin user created (password: 3030)');
  }
}

app.get('/', (req, res) => {
  res.render('login', { baseurl: '', title: 'Login', error: null });
});

// --- MCP (Model Context Protocol) remote endpoint ---

const mcpTools = [
  {
    name: "hive_find",
    description: "Vector similarity search (requires auth via username/password)",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Text query" },
        dbName: { type: "string", description: "Database to search" },
        topK: { type: "number", description: "Number of results (default: 10)" },
        username: { type: "string", description: "Username" },
        password: { type: "string", description: "Password" }
      },
      required: ["query", "dbName", "username", "password"]
    }
  },
  {
    name: "hive_add_file",
    description: "Add a file to a database (requires write access)",
    inputSchema: {
      type: "object",
      properties: {
        filePath: { type: "string", description: "Path to the file" },
        dbName: { type: "string", description: "Database name" },
        username: { type: "string", description: "Username" },
        password: { type: "string", description: "Password" }
      },
      required: ["filePath", "dbName", "username", "password"]
    }
  },
  {
    name: "hive_auth",
    description: "Verify credentials and list accessible databases",
    inputSchema: {
      type: "object",
      properties: {
        username: { type: "string" },
        password: { type: "string" }
      },
      required: ["username", "password"]
    }
  }
];

function mcpOk(data) {
  return { content: [{ type: "text", text: JSON.stringify(data, null, 2) }] };
}

function mcpErr(msg) {
  return { content: [{ type: "text", text: msg }], isError: true };
}

const mcpServer = new Server(
  { name: "hive-mcp", version: "1.4.0" },
  { capabilities: { tools: {} } }
);

mcpServer.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: mcpTools }));

mcpServer.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  try {
    // Ensure users are loaded from hive_users.json
    loadGlobalUsers();

    switch (name) {
      case "hive_find": {
        const { query, dbName, topK, username, password } = args;
        if (!dbName) return mcpErr("dbName required");
        Hive.auth(username, password);
        if (!userCanAccessDb(username, dbName)) return mcpErr(`Access denied to "${dbName}"`);
        const binPath = dbBinPath(dbName);
        if (!fs.existsSync(binPath)) return mcpErr(`Database "${dbName}" not found`);
        const index = await Hive.getIndex(binPath, dbName);
        const queryVec = await Hive.embed(query, 'text');
        const results = Hive.searchHybrid(index, queryVec, query, topK || 10)
          .filter(r => r.document.meta?.type !== 'image');
        return mcpOk(results.map(r => ({
          id: r.document.id, similarity: r.similarity, keywordScore: r.keywordScore ?? null, meta: r.document.meta, db: dbName
        })));
      }
      case "hive_add_file": {
        const { filePath, dbName, username, password } = args;
        if (!dbName) return mcpErr("dbName required");
        if (!filePath || !fs.existsSync(filePath)) return mcpErr(`File not found: ${filePath}`);
        Hive.auth(username, password);
        if (!userCanWriteDb(username, dbName)) return mcpErr(`Write access denied to "${dbName}"`);
        const binPath = path.join(DB_DIR, dbName, `${dbName}.bin`);
        if (!fs.existsSync(binPath)) return mcpErr(`Database "${dbName}" not found`);
        Hive.collections.clear();
        Hive.dbName = dbName;
        Hive.pathToDB = binPath;
        Hive.logging = false;
        await Hive.init({ pathToDB: binPath, dbName, logging: false });
        Hive.users.clear();
        await Hive.addFile(filePath);
        await Hive.saveToDisk();
        return mcpOk({ message: `File added to "${dbName}"` });
      }
      case "hive_auth": {
        Hive.auth(args.username, args.password);
        const dbs = getUserDatabases(args.username);
        const user = Hive.users.get(args.username);
        return mcpOk({ authenticated: true, username: args.username, roles: user.roles, databases: dbs });
      }
      default:
        return mcpErr(`Unknown tool: ${name}`);
    }
  } catch (e) {
    return mcpErr(e.message);
  }
});

// SSE transport for MCP — one connection at a time
let mcpSseTransport = null;

app.get('/mcp/sse', async (req, res) => {
  mcpSseTransport = new SSEServerTransport('/mcp/message', res);
  await mcpServer.connect(mcpSseTransport);
});

app.post('/mcp/message', async (req, res) => {
  if (mcpSseTransport) {
    await mcpSseTransport.handlePostMessage(req, res);
  }
});

app.get('/admin', loggedIn, requireAdmin, (req, res) => {
  res.sendFile(path.join(__dirname, 'admin', 'index.html'));
});

app.use('/admin', express.static(path.join(__dirname, 'admin')));
app.use(express.static(path.join(__dirname, 'admin')));

app.use((err, req, res, next) => {
  console.error('API error:', err.message);
  res.status(500).json({ error: err.message || 'Internal server error' });
});

app.listen(PORT, async () => {
  console.log(`Hive Admin: http://localhost:${PORT}`);
  try {
    await Hive.init({ logging: false, pathToDB: path.join(DB_DIR, 'Main', 'Main.bin') });
    await initHive();
  } catch (e) {
    console.error('Hive init error:', e.message);
  }
});
