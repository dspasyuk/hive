#!/usr/bin/env node
import { pipeline } from "@xenova/transformers";
import fs from "fs";
import path from "path";
import doc2txt from "./doc2txt.js";
import { fileURLToPath } from "url";
import readline from "readline";
import { Packr, UnpackrStream } from "msgpackr";
import crypto from "crypto";

/**
 * Copyright Denis Spasyuk
 * The Hive class is a database management system that provides a simple and efficient way to store and retrieve data.
 * Updated to support Hybrid Search (Cosine + Reranking) and Scientific Data.
 */

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Shared, byte-bounded LRU cache of decoded database indexes. Indexes are
// read-only projections of a .bin file (metadata + Float32 vectors + counts),
// keyed by path and invalidated when the file's mtime/size change.
const INDEX_CACHE_MAX_BYTES = Number(process.env.HIVE_INDEX_CACHE_BYTES || 1024 * 1024 * 1024);
const indexCache = new Map();
const indexCacheInflight = new Map();
const keywordCache = new WeakMap();
let indexCacheBytes = 0;

class Hive {
  static collections = new Map();
  static pipelines = {};
  static saveTimeout = null;
  static isSaving = false;

  // Default configuration
  static dbName = "Documents";
  static pathToDB = path.join(process.cwd(), "db", "Documents", "Documents.bin");
  static pathToDocs = false;
  static type = "text"; 
  static watch = false;
  static logging = false;
  static overlap = 0; // Default overlap
  static escapechar = false;
  // NEW: Reranking Configuration
  static useRerank = false;


  static users = new Map();
  static currentUser = null;
  static roleActions = {
    root: new Set(["*"]),
    dbOwner: new Set(["*"]),
    readWrite: new Set(["find", "insert", "update", "delete", "embed"]),
    read: new Set(["find", "embed"]),
    dbAdmin: new Set(["admin"])
  };

  static documents = {
    text: [".txt", ".doc", ".docx", ".pdf", ".md", ".py", ".js", ".sh", ".csv", ".json"],
    image: [".png", ".jpg", ".jpeg"],
  };
  static SliceSize = 512;
  static minSliceSize = 100;
  static saveInterval = 5000;
  static embedBatchSize = 32;
  
  // Updated models for better accuracy
  static models = {
    text: "Xenova/bge-base-en-v1.5", // Upgraded from MiniLM for better retrieval
    image: "Xenova/clip-vit-base-patch32", //Xenova/siglip-base-patch16-224 clip-vit-base-patch32 Upgraded from CLIP for better feature extraction
    rerank: "Xenova/ms-marco-MiniLM-L-6-v2",
  };
  
  static TransOptions = { pooling: "mean", normalize: false };

  /**
   * Initialize Hive DB with support for both text and image embeddings + Reranking
   * @param {Object} options
   */
  static async init(options = {}) {
    Hive.dbName = options.dbName || Hive.dbName;
    
    if (options.pathToDB) {
      const ext = path.extname(options.pathToDB);
      if (!ext) {
        Hive.pathToDB = path.join(options.pathToDB, Hive.dbName, Hive.dbName + ".bin");
        Hive.log(`pathToDB "${options.pathToDB}" is a directory, storing at ${Hive.pathToDB}`);
      } else {
        Hive.pathToDB = options.pathToDB.endsWith('.json') ? options.pathToDB.replace(/\.json$/, '.bin') : options.pathToDB;
      }
    } else if (options.storageDir) {
      Hive.pathToDB = path.join(options.storageDir, Hive.dbName, Hive.dbName + ".bin");
    } else {
      Hive.pathToDB = path.join(process.cwd(), "db", Hive.dbName, Hive.dbName + ".bin");
    }

    Hive.pathToDocs = options.pathToDocs !== undefined ? options.pathToDocs : Hive.pathToDocs;
    Hive.type = options.type || Hive.type;
    Hive.watch = options.watch !== undefined ? options.watch : Hive.watch;
    Hive.logging = options.logging !== undefined ? options.logging : Hive.logging;
    Hive.documents = options.documents || Hive.documents;
    Hive.SliceSize = options.SliceSize || Hive.SliceSize;
    Hive.minSliceSize = options.minSliceSize || Hive.minSliceSize;

    // NEW: Enable Re-ranking
    Hive.useRerank = options.rerank !== undefined ? options.rerank : false;

    Hive.escapechar = options.escapechar !== undefined ? options.escapechar : Hive.escapechar;

    // NEW: Allow custom models
    if (options.models) {
        Hive.models = { ...Hive.models, ...options.models };
    }
    
    // Handle overlap option
    if (options.overlap !== undefined) {
      Hive.overlap = options.overlap;
    } else {
      // Default to 5% if not specified
      Hive.overlap = Math.floor(Hive.SliceSize * 0.05);
    }
    Hive.createCollection(Hive.dbName);
    await Hive.loadToMemory();

    if (options.permissions && options.permissions.users) {
      for (const [username, config] of Object.entries(options.permissions.users)) {
        if (!Hive.users.has(username)) {
          const salt = Hive.generateSalt();
          const password = Hive.hashPassword(config.password, salt);
          Hive.users.set(username, {
            password,
            salt,
            roles: config.roles || ["read"]
          });
        }
      }
      if (options.permissions.autoAuth) {
        Hive.auth(options.permissions.autoAuth, options.permissions.users[options.permissions.autoAuth]?.password);
      }
    }

    await Hive.initTransformers();

    if (Hive.pathToDocs) {
      if (fs.existsSync(Hive.pathToDocs)) {
        if (!Hive.databaseExists()) {
          await Hive.pullDocuments(Hive.pathToDocs);
        } else {
          if (Hive.watch) {
            Hive.watchDocuments(Hive.pathToDocs);
          }
        }
      } else {
        Hive.log(`Document folder "${Hive.pathToDocs}" does not exist`);
      }
    } else {
      Hive.log(`Document folder not defined`);
    }
  }

  static databaseExists() {
    if (fs.existsSync(Hive.pathToDB)) {
        try {
            const stats = fs.statSync(Hive.pathToDB);
            return stats.size > 200;
        } catch (e) {
            return false;
        }
    }
    return false;
  }

  /**
   * Initialize transformers pipelines (Embedding + Reranker)
   */
  static async initTransformers() {
    if (!Hive.pipelines.text) {
        Hive.log("Initializing Text Model...");
        Hive.pipelines.text = await Hive.textEmbeddingInit();
    }
    if (!Hive.pipelines.image) {
        Hive.log("Initializing Image Model...");
        Hive.pipelines.image = await Hive.imageEmbeddingInit();
    }
    // NEW: Initialize Reranker if enabled
    if (Hive.useRerank && !Hive.pipelines.reranker) {
        Hive.log("Initializing Re-ranker...");
        Hive.pipelines.reranker = await pipeline("text-classification", Hive.models.rerank);
    }
  }
  static async getVector(input, options) {
      let type = Hive.type || 'text';
      
      // Auto-detect image type from extension if input is a string path
      if (typeof input === 'string') {
          const ext = path.extname(input).toLowerCase();
          if (Hive.documents.image.includes(ext)) {
              type = 'image';
          }
      }

      const vector = await Hive.embed(input, type);
      return { data: vector };
  }
  static log(...args) {
    if (Hive.logging) {
      console.log(...args);
    }
  }

  static async textEmbeddingInit() {
    return await pipeline("feature-extraction", Hive.models.text);
  }

  static async imageEmbeddingInit() {
    return await pipeline("image-feature-extraction", Hive.models.image);
  }

  static createCollection(name = Hive.dbName) {
    if (!Hive.collections.has(name)) {
      Hive.collections.set(name, []);
    }
  }

  static hashPassword(password, salt) {
    return crypto.createHash("sha256").update(salt + password).digest("hex");
  }

  static generateSalt() {
    return crypto.randomBytes(16).toString("hex");
  }

  static auth(username, password) {
    if (!Hive.users.has(username)) {
      throw new Error(`User "${username}" not found`);
    }
    const user = Hive.users.get(username);
    const hash = Hive.hashPassword(password, user.salt);
    if (hash !== user.password) {
      throw new Error("Invalid credentials");
    }
    Hive.currentUser = username;
    Hive.log(`User "${username}" authenticated`);
    return true;
  }

  static logout() {
    Hive.currentUser = null;
    Hive.log("User logged out");
  }

  static whoAmI() {
    return Hive.currentUser;
  }

  static checkAuth(action) {
    if (Hive.users.size === 0) return;
    if (!Hive.currentUser) {
      throw new Error("Not authenticated. Call Hive.auth(username, password) first.");
    }
    const user = Hive.users.get(Hive.currentUser);
    if (!user) {
      throw new Error(`Current user "${Hive.currentUser}" not found`);
    }
    for (const role of user.roles) {
      const actions = Hive.roleActions[role];
      if (!actions) continue;
      if (actions.has("*") || actions.has(action)) {
        return;
      }
    }
    throw new Error(`User "${Hive.currentUser}" is not authorized to perform "${action}"`);
  }

  static createUser({ user, pwd, roles = ["read"] }) {
    Hive.checkAuth("admin");
    if (Hive.users.has(user)) {
      throw new Error(`User "${user}" already exists`);
    }
    const salt = Hive.generateSalt();
    const password = Hive.hashPassword(pwd, salt);
    Hive.users.set(user, { password, salt, roles });
    Hive.saveToDisk();
    Hive.log(`User "${user}" created with roles: ${roles.join(", ")}`);
    return true;
  }

  static dropUser(username) {
    Hive.checkAuth("admin");
    if (!Hive.users.has(username)) {
      throw new Error(`User "${username}" not found`);
    }
    if (Hive.currentUser === username) {
      throw new Error("Cannot delete yourself");
    }
    Hive.users.delete(username);
    Hive.saveToDisk();
    Hive.log(`User "${username}" dropped`);
    return true;
  }

  static getUsers() {
    Hive.checkAuth("admin");
    const result = [];
    for (const [username, data] of Hive.users) {
      result.push({ user: username, roles: data.roles });
    }
    return result;
  }

  static randomId() {
    return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
  }

  static deleteOne(id) {
    Hive.checkAuth("delete");
    if (Hive.collections.has(Hive.dbName)) {
      const collection = Hive.collections.get(Hive.dbName);
      const filteredCollection = collection.filter((item) => item.id !== id);
      Hive.collections.set(Hive.dbName, filteredCollection);
      Hive.saveToDisk();
    }
  }

  static insertOne(entry) {
    Hive.checkAuth("insert");
    if (Hive.collections.has(Hive.dbName)) {
      const { vector, meta } = entry;
      const magnitude = Hive.normalize(vector);
      Hive.log("Inserting entry:", entry.meta.title);
      Hive.collections.get(Hive.dbName).push({
        vector: Hive.toFloat32(vector),
        magnitude,
        meta,
        id: Hive.randomId(),
      });
      // Save handled by caller (import-folders.js)
    }
  }

  static updateOne(query, entry) {
    Hive.checkAuth("update");
    if (Hive.collections.has(Hive.dbName)) {
      Hive.findMeta(query, entry);
      const collection = Hive.collections.get(Hive.dbName);
      if (collection) keywordCache.delete(collection);
      Hive.saveToDisk();
    }
  }

  static insertMany(entries) {
    Hive.checkAuth("insert");
    if (Hive.collections.has(Hive.dbName)) {
      const collection = Hive.collections.get(Hive.dbName);
      for (let i = 0; i < entries.length; i++) {
        const { vector, meta } = entries[i];
        collection.push({
          vector: Hive.toFloat32(vector),
          meta,
          magnitude: Hive.normalize(vector),
          id: Hive.randomId(),
        });
      }
      Hive.saveToDisk();
    }
  }

  static async ensureDirectoryExists(pathToDB) {
    const dir = path.dirname(pathToDB);
    try {
        await fs.promises.access(dir);
    } catch {
        await fs.promises.mkdir(dir, { recursive: true });
    }
  }

  static saveToDisk() {
    const dbPath = Hive.pathToDB;
    return new Promise((resolve, reject) => {
      if (Hive.saveTimeout) clearTimeout(Hive.saveTimeout);
      Hive.saveTimeout = setTimeout(async () => {
        if (Hive.isSaving) {
          Hive._pendingSave = { resolve, reject, dbPath };
          return;
        }
        Hive.isSaving = true;
        try {
          await Hive.ensureDirectoryExists(dbPath);
          const tempPath = `${dbPath}.tmp`;
          const fileStream = fs.createWriteStream(tempPath, { flags: 'w' });

          let streamError = null;
          fileStream.on('error', (err) => { streamError = err; });

          const packr = new Packr({ structuredClone: true });

          if (Hive.users.size > 0) {
            const usersData = {};
            for (const [username, data] of Hive.users.entries()) {
              usersData[username] = data;
            }
            const record = { collection: "__hive__", _type: "users", data: usersData };
            if (!fileStream.write(packr.pack(record))) {
              await new Promise(r => fileStream.once('drain', r));
            }
          }

          for (const [key, value] of Hive.collections.entries()) {
            for (const entry of value) {
              const record = {
                collection: key, id: entry.id,
                vector: entry.vector, meta: entry.meta, magnitude: entry.magnitude
              };
              if (!fileStream.write(packr.pack(record))) {
                await new Promise(r => fileStream.once('drain', r));
              }
            }
          }

          fileStream.end();
          await new Promise((res, rej) => {
            fileStream.on('finish', res);
            fileStream.on('error', rej);
          });
          if (streamError) throw streamError;
          await fs.promises.rename(tempPath, dbPath);
          Hive.log(`Database saved to ${dbPath}`);
          resolve();
        } catch (error) {
          console.error('Error saving database:', error.message);
          reject(error);
        } finally {
          Hive.isSaving = false;
          if (Hive._pendingSave) {
            const p = Hive._pendingSave;
            Hive._pendingSave = null;
            Hive.saveToDisk().then(p.resolve).catch(p.reject);
          }
        }
      }, Hive.saveInterval);
    });
  }

  static async loadToMemory() {
    let loadPath = Hive.pathToDB;

    // Track if we loaded from a legacy file
    let loadedFromLegacy = false;
    let legacyFilePath = null;

    if (!fs.existsSync(loadPath)) {
        if (loadPath.endsWith('.bin')) {
            const legacyPath = loadPath.replace(/\.bin$/, '.json');
            if (fs.existsSync(legacyPath)) {
                loadPath = legacyPath;
                legacyFilePath = legacyPath;
                loadedFromLegacy = true;
                Hive.log(`No .bin database found. Falling back to legacy JSON at ${legacyPath}`);
            } else {
                return;
            }
        } else {
            return;
        }
    }

    if (Hive.collections.size === 0 || Hive.collections.get(Hive.dbName)?.length === 0) {
      try {
        Hive.collections.clear();

        let isJson = loadPath.endsWith('.json');
        if (fs.existsSync(loadPath)) {
            const fd = fs.openSync(loadPath, 'r');
            const buf = Buffer.alloc(1);
            if (fs.readSync(fd, buf, 0, 1, null) > 0) {
                if (buf[0] >= 0x80) {
                    isJson = false;
                }
            }
            fs.closeSync(fd);
        }

        const fileStream = fs.createReadStream(loadPath);

        if (isJson) {
            const rl = readline.createInterface({
                input: fileStream,
                crlfDelay: Infinity
            });

            for await (const line of rl) {
                if (!line.trim()) continue;
                try {
                    const record = JSON.parse(line);
                    
                    if (record && !record.collection && !record.vector && typeof record === 'object') {
                         for (const [dbName, entries] of Object.entries(record)) {
                            Hive.createCollection(dbName);
                            const collection = Hive.collections.get(dbName);
                            for (const entry of entries) {
                                collection.push({
                                    vector: new Float32Array(entry.vector),
                                    meta: entry.meta,
                                    magnitude: entry.magnitude,
                                    id: entry.id,
                                });
                            }
                         }
                         continue;
                    }

                    if (record.collection) {
                        if (!Hive.collections.has(record.collection)) {
                            Hive.createCollection(record.collection);
                        }
                        Hive.collections.get(record.collection).push({
                            vector: new Float32Array(record.vector),
                            meta: record.meta,
                            magnitude: record.magnitude,
                            id: record.id,
                        });
                    }
                } catch (e) {
                    console.warn("Skipping invalid line in DB:", e.message);
                }
            }
        } else {
            const unpacker = new UnpackrStream({ structuredClone: true });
            fileStream.pipe(unpacker);
            
            for await (const record of unpacker) {
                try {
                    if (record.collection) {
                        if (!Hive.collections.has(record.collection)) {
                            Hive.createCollection(record.collection);
                        }
                        Hive.collections.get(record.collection).push({
                            vector: record.vector instanceof Float32Array ? record.vector : new Float32Array(record.vector),
                            meta: record.meta,
                            magnitude: record.magnitude,
                            id: record.id,
                        });
                    }
                } catch (e) {
                    console.warn("Skipping invalid entry in DB:", e.message);
                }
            }
        }
        
        Hive.log(`Database loaded into memory from ${loadPath}`);

        if (loadedFromLegacy && legacyFilePath) {
            Hive.log(`Initiating automatic migration of legacy JSON database to msgpackr binary format.`);
            // Force a save to disk to create the .bin file immediately
            Hive.saveToDisk();
            
            // Rename the old json file to .bak to avoid confusion
            try {
                fs.renameSync(legacyFilePath, legacyFilePath + ".bak");
                Hive.log(`Renamed legacy database to ${legacyFilePath}.bak`);
            } catch (e) {
                console.warn("Could not rename legacy database file:", e.message);
            }
        }

        if (Hive.collections.has("__hive__")) {
          const systemEntries = Hive.collections.get("__hive__");
          for (const entry of systemEntries) {
            if (entry._type === "users" && entry.data) {
              Hive.users.clear();
              for (const [username, data] of Object.entries(entry.data)) {
                Hive.users.set(username, data);
              }
            }
          }
          Hive.collections.delete("__hive__");
        }
      } catch (error) {
        console.error("Error loading database:", error);
      }
    }
  }

  static findMeta(query, entry) {
    const collection = Hive.collections.get(Hive.dbName);
    if (!collection) return;
    
    for (let i = 0; i < collection.length; i++) {
      const item = collection[i];
      if (item.meta.filePath === query.filePath) {
        collection[i] = entry;
        break;
      }
    }
  }

  /**
   * Stream-decode a .bin file into a read-only index: entries (metadata +
   * Float32 vectors + precomputed magnitudes), per-collection counts, and
   * dimension summary. Does not touch Hive.collections or load models.
   */
  static buildIndex(binPath, defaultCollection) {
    return new Promise((resolve, reject) => {
      const entries = [];
      const collections = new Map();
      let bytes = 0;
      if (!fs.existsSync(binPath)) {
        return resolve({ entries, collections, count: 0, bytes: 0, dims: [] });
      }
      const readStream = fs.createReadStream(binPath);
      const unpacker = new UnpackrStream({ structuredClone: true });
      readStream.pipe(unpacker)
        .on('data', (rec) => {
          if (!rec || typeof rec !== 'object') return;
          const coll = rec.collection || defaultCollection || 'default';
          const vec = rec.vector && rec.vector.length ? Hive.toFloat32(rec.vector) : null;
          const entry = {
            id: rec.id,
            collection: coll,
            meta: rec.meta || {},
            vector: vec,
            magnitude: vec
              ? (typeof rec.magnitude === 'number' && rec.magnitude ? rec.magnitude : Hive.normalize(vec))
              : 0,
          };
          if (rec._type !== undefined) entry._type = rec._type;
          if (rec.data !== undefined) entry.data = rec.data;
          entries.push(entry);
          bytes += (vec ? vec.byteLength : 0) + 96
            + (entry.meta.content ? entry.meta.content.length * 2 : 0)
            + (entry.meta.title ? entry.meta.title.length * 2 : 0)
            + (entry.meta.filePath ? entry.meta.filePath.length * 2 : 0);
          if (coll !== '__hive__') {
            let c = collections.get(coll);
            if (!c) { c = { count: 0, types: new Set() }; collections.set(coll, c); }
            c.count++;
            if (entry.meta.type) c.types.add(entry.meta.type);
          }
        })
        .on('end', () => {
          let count = 0;
          const dims = new Set();
          for (const e of entries) {
            if (e.collection === '__hive__') continue;
            count++;
            if (e.vector) dims.add(e.vector.length);
          }
          resolve({ entries, collections, count, bytes, dims: [...dims] });
        })
        .on('error', reject);
    });
  }

  /**
   * Return a cached index for a .bin file, rebuilding it only when the file's
   * mtime/size change. Concurrent callers share one build.
   */
  static async getIndex(binPath, defaultCollection) {
    let st;
    try { st = fs.statSync(binPath); } catch {
      return { entries: [], collections: new Map(), count: 0, bytes: 0, dims: [] };
    }

    const hit = indexCache.get(binPath);
    if (hit && hit.mtimeMs === st.mtimeMs && hit.size === st.size) {
      indexCache.delete(binPath);
      indexCache.set(binPath, hit);
      return hit.index;
    }

    let pending = indexCacheInflight.get(binPath);
    if (!pending) {
      pending = Hive.buildIndex(binPath, defaultCollection);
      indexCacheInflight.set(binPath, pending);
      const clear = () => {
        if (indexCacheInflight.get(binPath) === pending) indexCacheInflight.delete(binPath);
      };
      pending.then(clear, clear);
    }
    const index = await pending;

    const existing = indexCache.get(binPath);
    if (existing && existing.mtimeMs === st.mtimeMs && existing.size === st.size) {
      return existing.index;
    }
    if (existing) indexCacheBytes -= existing.index.bytes;
    indexCache.set(binPath, { mtimeMs: st.mtimeMs, size: st.size, index });
    indexCacheBytes += index.bytes;
    Hive._evictIndexCache(binPath);
    return index;
  }

  static _evictIndexCache(pinned) {
    for (const [key, value] of indexCache) {
      if (indexCacheBytes <= INDEX_CACHE_MAX_BYTES) break;
      if (key === pinned) continue;
      indexCache.delete(key);
      indexCacheBytes -= value.index.bytes;
    }
  }

  static invalidateIndex(binPath) {
    const hit = indexCache.get(binPath);
    if (hit) {
      indexCache.delete(binPath);
      indexCacheBytes -= hit.index.bytes;
    }
  }

  static clearIndexCache() {
    indexCache.clear();
    indexCacheInflight.clear();
    indexCacheBytes = 0;
  }

  static indexCacheStats() {
    return { entries: indexCache.size, bytes: indexCacheBytes, maxBytes: INDEX_CACHE_MAX_BYTES };
  }

  static tokenize(text) {
    if (typeof text !== "string" || !text) return [];
    return text.toLowerCase().match(/[\p{L}\p{N}_]+/gu) || [];
  }

  /**
   * Build an in-memory BM25 keyword index over an index's entries. Titles are
   * weighted more heavily than content so exact title/name matches rank well.
   */
  static buildKeywordIndex(entries, options = {}) {
    const k1 = options.k1 ?? 1.2;
    const b = options.b ?? 0.75;
    const titleWeight = options.titleWeight ?? 4;
    const contentWeight = options.contentWeight ?? 1;
    const docs = [];
    const lengths = [];
    const terms = new Map();
    const postings = new Map();
    let totalLength = 0;

    const addTerm = (tf, token, weight) => {
      let tid = terms.get(token);
      if (tid === undefined) {
        tid = terms.size;
        terms.set(token, tid);
      }
      tf.set(tid, (tf.get(tid) || 0) + weight);
    };

    for (const entry of entries) {
      if (!entry || entry.collection === "__hive__") continue;
      const titleTokens = Hive.tokenize(entry.meta?.title);
      const contentTokens = Hive.tokenize(entry.meta?.content);
      if (!titleTokens.length && !contentTokens.length) continue;

      const tf = new Map();
      for (const token of titleTokens) addTerm(tf, token, titleWeight);
      for (const token of contentTokens) addTerm(tf, token, contentWeight);

      let length = 0;
      for (const value of tf.values()) length += value;
      if (!length) continue;

      const docIndex = docs.length;
      docs.push(entry);
      lengths.push(length);
      totalLength += length;

      for (const [tid, freq] of tf) {
        let posting = postings.get(tid);
        if (!posting) {
          posting = [];
          postings.set(tid, posting);
        }
        posting.push(docIndex, freq);
      }
    }

    const N = docs.length;
    return {
      docs,
      lengths,
      terms,
      postings,
      N,
      avgdl: N ? totalLength / N : 0,
      k1,
      b,
    };
  }

  static getKeywordIndex(index) {
    if (!index) return null;
    if (!index.keyword) index.keyword = Hive.buildKeywordIndex(index.entries || []);
    return index.keyword;
  }

  static getCollectionKeywordIndex(entries) {
    let hit = keywordCache.get(entries);
    if (!hit || hit.length !== entries.length) {
      hit = { length: entries.length, index: Hive.buildKeywordIndex(entries) };
      keywordCache.set(entries, hit);
    }
    return hit.index;
  }

  static searchKeyword(index, query, topK = 10) {
    if (!index) return [];
    const kw = index.keyword || Hive.getKeywordIndex(index);
    if (!kw || !kw.N || !query) return [];
    const queryTokens = Hive.tokenize(query);
    if (!queryTokens.length) return [];

    const scores = new Map();
    const seenTerms = new Set();
    for (const token of queryTokens) {
      const tid = kw.terms.get(token);
      if (tid === undefined || seenTerms.has(tid)) continue;
      seenTerms.add(tid);
      const posting = kw.postings.get(tid);
      if (!posting) continue;
      const df = posting.length / 2;
      const idf = Math.log(1 + (kw.N - df + 0.5) / (df + 0.5));
      for (let i = 0; i < posting.length; i += 2) {
        const docIndex = posting[i];
        const tf = posting[i + 1];
        const dl = kw.lengths[docIndex];
        const denom = tf + kw.k1 * (1 - kw.b + kw.b * (dl / kw.avgdl));
        const score = idf * ((tf * (kw.k1 + 1)) / denom);
        scores.set(docIndex, (scores.get(docIndex) || 0) + score);
      }
    }

    if (!scores.size) return [];
    const ranked = [];
    for (const [docIndex, score] of scores) {
      ranked.push({ document: kw.docs[docIndex], keywordScore: score });
    }
    ranked.sort((a, b) => b.keywordScore - a.keywordScore);
    return ranked.slice(0, Math.max(1, topK | 0));
  }

  /**
   * Reciprocal Rank Fusion. Ranks from each list are combined as
   * weight / (k + rank + 1), which is robust to differently scaled scores.
   */
  static rrfFuse(lists, options = {}) {
    const k = options.k ?? 60;
    const weights = options.weights || [];
    const labels = options.labels || [];
    const fused = new Map();

    for (let li = 0; li < lists.length; li++) {
      const list = lists[li];
      if (!list || !list.length) continue;
      const weight = weights[li] ?? 1;
      const label = labels[li] ?? li;
      for (let rank = 0; rank < list.length; rank++) {
        const item = list[rank];
        const doc = item.document || item;
        if (!doc) continue;
        const key = `${doc._db || ""}:${doc.id}`;
        let rec = fused.get(key);
        if (!rec) {
          rec = { document: doc, similarity: null, keywordScore: null, rrfScore: 0, sources: [] };
          fused.set(key, rec);
        }
        rec.rrfScore += weight / (k + rank + 1);
        if (item.similarity != null && (rec.similarity == null || item.similarity > rec.similarity)) {
          rec.similarity = item.similarity;
        }
        if (item.keywordScore != null && (rec.keywordScore == null || item.keywordScore > rec.keywordScore)) {
          rec.keywordScore = item.keywordScore;
        }
        if (!rec.sources.includes(label)) rec.sources.push(label);
      }
    }

    return [...fused.values()].sort((a, b) => b.rrfScore - a.rrfScore);
  }

  static normalizeRrfScores(results) {
    if (!results.length) return results;
    const max = results[0].rrfScore || 0;
    for (const result of results) {
      result.score = max ? result.rrfScore / max : 0;
    }
    return results;
  }

  static searchHybrid(index, queryVec, queryText, topK = 10, options = {}) {
    const k = Math.max(1, topK | 0);
    const candidateK = Math.max(k, options.candidateK || k * 4);
    const semantic = Hive.searchVector(index, queryVec, candidateK);
    if (!queryText) return semantic.slice(0, k);

    const keyword = Hive.searchKeyword(index, queryText, candidateK);
    if (!keyword.length) return semantic.slice(0, k);
    if (!semantic.length) return keyword.slice(0, k);

    return Hive.rrfFuse([semantic, keyword], {
      k: options.rrfK ?? 60,
      weights: options.weights,
    }).slice(0, k);
  }

  /**
   * Cosine search over an index (or any { entries } collection) with a
   * precomputed query vector. Uses a bounded min-heap so only the current
   * top-K candidates are tracked — no per-entry allocation, no full sort.
   */
  static searchVector(index, queryVec, topK = 10) {
    const q = Hive.toFloat32(queryVec);
    if (!q) return [];
    const qmag = Hive.normalize(q);
    if (!qmag) return [];
    const k = Math.max(1, topK | 0);
    const qLen = q.length;
    const heap = [];
    for (const e of index.entries) {
      const v = e.vector;
      if (!v || v.length !== qLen) continue;
      let dot = 0;
      for (let i = 0; i < qLen; i++) dot += q[i] * v[i];
      const similarity = dot / (qmag * (e.magnitude || 1));
      if (heap.length < k) {
        heap.push({ document: e, similarity });
        Hive._heapSiftUp(heap, heap.length - 1);
      } else if (similarity > heap[0].similarity) {
        heap[0] = { document: e, similarity };
        Hive._heapSiftDown(heap, 0, heap.length);
      }
    }
    return heap.sort((a, b) => b.similarity - a.similarity);
  }

  static _heapSiftUp(heap, i) {
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (heap[parent].similarity <= heap[i].similarity) break;
      [heap[parent], heap[i]] = [heap[i], heap[parent]];
      i = parent;
    }
  }

  static _heapSiftDown(heap, i, size) {
    for (;;) {
      const left = 2 * i + 1;
      const right = left + 1;
      let smallest = i;
      if (left < size && heap[left].similarity < heap[smallest].similarity) smallest = left;
      if (right < size && heap[right].similarity < heap[smallest].similarity) smallest = right;
      if (smallest === i) break;
      [heap[smallest], heap[i]] = [heap[i], heap[smallest]];
      i = smallest;
    }
  }

  /**
   * Find similar vectors with optional Re-ranking
   * @param {Array|string} queryInput - Vector or text
   * @param {number} topK 
   * @returns {Array}
   */
  static async find(queryInput, topK = 10) {
    Hive.checkAuth("find");
    let queryVector = queryInput;
    let queryText = "";

    // Auto-detect if input is text string, store it for re-ranking
    if (typeof queryInput === "string") {
        queryText = queryInput;
        queryVector = await Hive.embed(queryInput, "text");
    }

    const collection = Hive.collections.get(Hive.dbName) || [];
    const shouldRerank = Hive.useRerank && Hive.pipelines.reranker && queryText;

    // 1. Initial Retrieval (hybrid dense + BM25 fused with RRF)
    const pool = shouldRerank ? Math.max(topK * 4, 50) : topK;
    const searchIndex = queryText
      ? { entries: collection, keyword: Hive.getCollectionKeywordIndex(collection) }
      : { entries: collection };
    const results = Hive.searchHybrid(searchIndex, queryVector, queryText, pool);

    // 2. Cross-encoder reranking over the candidate pool
    if (shouldRerank && results.length) {
        try {
            const docTexts = results.map((res) =>
                res.document.meta?.content ? res.document.meta.content.slice(0, 1000) : ""
            );
            const scores = await Hive.rerank(queryText, docTexts);
            const reranked = results.map((res, i) => ({
                ...res,
                similarity: scores[i] ?? res.similarity,
                original_similarity: res.similarity,
            }));
            reranked.sort((a, b) => b.similarity - a.similarity);
            return reranked.slice(0, topK);
        } catch (err) {
            console.error("Rerank error:", err.message);
        }
    }

    return results.slice(0, topK);
  }

  /**
   * Cross-encoder relevance scores for query/document pairs. Uses the
   * reranker pipeline's tokenizer/model directly so `text_pair` is honored,
   * and processes pairs in batches.
   */
  static async rerank(queryText, docTexts, options = {}) {
    const pipe = Hive.pipelines.reranker;
    if (!pipe || !pipe.tokenizer || !pipe.model || !queryText || !docTexts.length) {
      return docTexts.map(() => null);
    }
    const batchSize = Math.max(1, options.batchSize || 8);
    const maxChars = options.maxChars || 1000;
    const scores = new Array(docTexts.length).fill(null);

    for (let start = 0; start < docTexts.length; start += batchSize) {
      const docs = docTexts.slice(start, start + batchSize).map((t) => (t || "").slice(0, maxChars));
      const queries = new Array(docs.length).fill(queryText);
      const inputs = pipe.tokenizer(queries, {
        text_pair: docs,
        padding: true,
        truncation: true,
      });
      const output = await pipe.model(inputs);
      const logits = output.logits;
      const rows = logits.dims[0];
      const labels = logits.dims.length > 1 ? logits.dims[logits.dims.length - 1] : 1;
      for (let i = 0; i < rows && i < docs.length; i++) {
        scores[start + i] = labels > 1
          ? logits.data[i * labels + (labels - 1)]
          : logits.data[i];
      }
    }
    return scores;
  }

  static cosineSimilarity(queryVector, itemVector, queryVectorMag, itemVectorMag) {
    let dotProduct = 0;
    for (let i = 0; i < queryVector.length; i++) {
      dotProduct += queryVector[i] * itemVector[i];
    }
    return dotProduct / (queryVectorMag * itemVectorMag);
  }

  static normalize(vector) {
    let sum = 0;
    for (let i = 0; i < vector.length; i++) {
      sum += vector[i] * vector[i];
    }
    return Math.sqrt(sum);
  }

  /**
   * Coerce an embedding result into a Float32Array so vectors are stored
   * compactly (4 bytes/dim) instead of as boxed JS number arrays.
   */
  static toFloat32(vector) {
    if (vector == null) return vector;
    if (vector instanceof Float32Array) return vector;
    if (Array.isArray(vector)) return Float32Array.from(vector);
    if (ArrayBuffer.isView(vector)) {
      return new Float32Array(vector.buffer, vector.byteOffset, Math.floor(vector.byteLength / 4));
    }
    return vector;
  }

  static embeddingText(title, chunk) {
    return title ? `${title}\n${chunk}` : chunk;
  }

  static buildTextEntry(vector, chunk, filePath, title = "") {
    const fv = Hive.toFloat32(vector);
    return {
      vector: fv,
      magnitude: Hive.normalize(fv),
      meta: {
        content: Hive.escapeChars(chunk),
        href: filePath,
        title: Hive.escapeChars(title || chunk.slice(0, 20)),
        filePath,
        type: "text",
      },
      id: Hive.randomId(),
    };
  }

  /**
   * Embed and index many text chunks in batches (one pipeline call per batch
   * instead of one per chunk) — the dominant cost when importing large files.
   */
  static async addTextChunks(chunks, filePath, title = "") {
    if (!chunks.length) return;
    if (!Hive.pipelines.text) await Hive.initTransformers();
    const collection = Hive.collections.get(Hive.dbName);
    if (!collection) return;
    const batchSize = Hive.embedBatchSize > 0 ? Hive.embedBatchSize : 32;

    for (let i = 0; i < chunks.length; i += batchSize) {
      const batch = chunks.slice(i, i + batchSize);
      let data, dim;
      const texts = batch.map((chunk) => Hive.embeddingText(title, chunk));
      try {
        const out = await Hive.pipelines.text(texts, Hive.TransOptions);
        data = out.data || out;
        if (out.dims && out.dims.length >= 2) {
          dim = out.dims[out.dims.length - 1];
        } else {
          dim = Math.floor(data.length / batch.length);
        }
      } catch (e) {
        // Fall back to one-by-one if the batch pipeline call fails
        for (const chunk of batch) {
          const single = await Hive.pipelines.text(Hive.embeddingText(title, chunk), Hive.TransOptions);
          collection.push(Hive.buildTextEntry(single.data || single, chunk, filePath, title));
        }
        continue;
      }
      for (let b = 0; b < batch.length; b++) {
        collection.push(Hive.buildTextEntry(data.slice(b * dim, (b + 1) * dim), batch[b], filePath, title));
      }
    }
  }

  static tokenCount(text) {
    const tokens = text.match(/\b\w+\b/g) || [];
    return [tokens, tokens.length];
  }

  static async addFile(filePath, filename, options = {}) {
    Hive.checkAuth("insert");
    const save = options.save !== false;
    try {
      const ext = path.extname(filePath).toLowerCase();
      if (Hive.documents.text.includes(ext)) {
          await Hive.readFile(filePath, "text", { title: filename || path.basename(filePath) });
          if (save) await Hive.saveToDisk();
      } else if (Hive.documents.image.includes(ext)) {
          await Hive.addItem("", filePath, "image", filename);
          if (save) await Hive.saveToDisk();
      }
    } catch (error) {
      console.error("Error adding file:", error);
    }
  }

  static async addItem(input, filePath = "", type = "text", filename = "") {
    try {
      let result;
      if (type === "text") {
        if (!Hive.pipelines.text) await Hive.initTransformers();
        result = await Hive.pipelines.text(input, Hive.TransOptions);
      } else if (type === "image") {
        if (!Hive.pipelines.image) await Hive.initTransformers();
        result = await Hive.pipelines.image(filePath, Hive.TransOptions);
      }
      
      const vectorData = result.data || result;
      
      // Use filename if provided, otherwise derive
      const actualTitle = filename || (type === "text" ? Hive.escapeChars(input.slice(0, 20)) : `Image: ${path.basename(filePath)}`);

      Hive.insertOne({
        vector: Hive.toFloat32(vectorData),
        meta: {
          content: type === "text" ? Hive.escapeChars(input) : `Image: ${path.basename(filePath)}`,
          href: filePath,
          title: actualTitle,
          filePath: filePath,
          type: type 
        },
      });
    } catch (error) {
      console.error("Error adding item:", error);
    }
  }

  static async pullDocuments(dir) {
    try {
        const files = await fs.promises.readdir(dir, { withFileTypes: true });
        for (const file of files) {
          let fullPath = path.join(dir, file.name);
          const ext = path.extname(file.name).toLowerCase();

          if (file.isSymbolicLink()) {
            try {
              fullPath = await fs.promises.readlink(fullPath);
            } catch (error) { continue; }
          }

          try {
            const stats = await fs.promises.stat(fullPath);
            if (stats.isDirectory()) {
              await Hive.pullDocuments(fullPath);
            } else if (Hive.documents.text.includes(ext) && !Hive.fileExistsInDatabase(fullPath)) {
              await Hive.readFile(fullPath, "text");
            } else if (Hive.documents.image.includes(ext) && !Hive.fileExistsInDatabase(fullPath)) {
              await Hive.readFile(fullPath, "image");
            }
          } catch (error) {
            console.error(`Error processing ${fullPath}:`, error.message);
          }
        }
    } catch (err) {
        console.error(`Error reading directory ${dir}:`, err.message);
    }
  }

  static async updateFile(filePath) {
    try {
      const ext = path.extname(filePath).toLowerCase();
      let type = "text";
      if (Hive.documents.image.includes(ext)) {
          type = "image";
      } else if (!Hive.documents.text.includes(ext)) {
          return;
      }

      let input = "";
      if (type === "text") {
        const extracted = await doc2txt.extractTextFromFile(filePath, { fast: true });
        input = extracted.text;
      } else {
          input = filePath; 
      }

      const title = path.basename(filePath);
      let result;
      if (type === "text") {
          result = await Hive.pipelines.text(Hive.embeddingText(title, input), Hive.TransOptions);
      } else {
          result = await Hive.pipelines.image(input, Hive.TransOptions);
      }

      const vectorData = result.data || result;
      
      const newEntry = { 
        vector: Hive.toFloat32(vectorData), 
        meta: { 
            filePath, 
            type,
            content: type === "text" ? Hive.escapeChars(input) : `Image: ${path.basename(filePath)}`,
            title: type === "text" ? Hive.escapeChars(title) : `Image: ${path.basename(filePath)}`,
            href: filePath
        },
        magnitude: Hive.normalize(vectorData),
        id: Hive.randomId()
      };
      Hive.updateOne({ filePath }, newEntry);
    } catch (error) {
      console.error(`Error updating file ${filePath}:`, error);
    }
  }

  static removeFile(filePath) {
    Hive.checkAuth("delete");
    const collection = Hive.collections.get(Hive.dbName);
    if (collection) {
      Hive.collections.set(
        Hive.dbName,
        collection.filter((item) => item.meta.filePath !== filePath)
      );
    }
  }

  static async watchDocuments(dir) {
    try {
      const chokidar = await import("chokidar");
      const watcher = chokidar.watch(dir, {
        ignored: (file, _stats) => {
            if (!_stats) return false;
            if (_stats.isDirectory()) return false;
            const ext = path.extname(file).toLowerCase();
            return !Hive.documents.text.includes(ext) && !Hive.documents.image.includes(ext);
        },
        persistent: true,
      });

      watcher
        .on("add", async (filePath) => {
          if (!(await Hive.fileExistsInDatabase(filePath))) {
             await Hive.updateFile(filePath);
          }
        })
        .on("change", async (filePath) => {
          await Hive.updateFile(filePath);
        })
        .on("unlink", async (filePath) => {
          Hive.removeFile(filePath);
          Hive.saveToDisk();
        });
    } catch (error) {
      console.error("Error setting up file watcher:", error);
    }
  }

  static fileExistsInDatabase(filePath) {
    Hive.checkAuth("find");
    const collection = Hive.collections.get(Hive.dbName);
    if (!collection) return false;
    return collection.some((item) => item.meta.filePath === filePath);
  }

  /**
   * Split text into overlapping token windows honoring SliceSize/overlap/
   * minSliceSize. Pure; reusable by bulk importers.
   */
  static chunkText(text, filePath = "") {
    const [tokens, len] = Hive.tokenCount(text);
    const chunks = [];
    let startIndex = 0;

    let overlapTokens = 0;
    if (Hive.overlap < 1 && Hive.overlap > 0) {
      overlapTokens = Math.floor(Hive.SliceSize * Hive.overlap);
    } else {
      overlapTokens = Math.floor(Hive.overlap);
    }

    const stride = Math.max(1, Hive.SliceSize - overlapTokens);

    while (startIndex < len) {
      const endIndex = Math.min(startIndex + Hive.SliceSize, len);
      const sliceLength = endIndex - startIndex;

      if (sliceLength >= Hive.minSliceSize) {
        chunks.push(tokens.slice(startIndex, endIndex).join(" "));
      } else if (sliceLength > 0 && Hive.minSliceSize > 1) {
        Hive.log(`Skipped ${sliceLength} token slice in ${filePath ? path.basename(filePath) : "text"} (minSliceSize=${Hive.minSliceSize})`);
      }

      if (endIndex === len) break;
      startIndex += stride;
    }
    return chunks;
  }

  static async readFile(filePath, type, options = {}) {
    try {
      Hive.log(`Processing: ${path.basename(filePath)}`);
      if (type === "text") {
        const { title, ...extractOptions } = options;
        const { text } = await doc2txt.extractTextFromFile(filePath, { fast: true, ...extractOptions });
        const chunks = Hive.chunkText(text, filePath);
        await Hive.addTextChunks(chunks, filePath, title || path.basename(filePath));
      } else if (type === "image") {
        await Hive.addItem("", filePath, type);
      }
    } catch (error) {
      console.error(`Error reading file ${filePath}:`, error.message);
    }
  }


  /**
   * Embed input using the specified type
   * @param {string} input - Text content or image path
   * @param {string} type - "text" or "image"
   * @returns {Promise<Array>}
   */
  static async embed(input, type = "text") {
      Hive.checkAuth("embed");
      if (!Hive.pipelines[type]) {
          await Hive.initTransformers();
      }
      const result = await Hive.pipelines[type](input, Hive.TransOptions);
      const vectorData = result.data || result;
      return Array.isArray(vectorData) ? vectorData : Array.from(vectorData);
  }

  /**
   * UPDATED: Relaxed Regex for Scientific Data
   * Preserves numbers, dots, hyphens, and percent signs.
   */
  static escapeChars(text) {
    if (!Hive.escapechar) return text;

    return (
      text
        .replace(/[^A-Za-z0-9\s.\-%]/g, " ") // Allow scientific notation parts
        .replace(/\s+/g, " ")
        .replace(/\b([A-Za-z])\b(\s+\1)+/g, "")
        .replace(/\b[A-Za-z]\b/g, "")
        .trim()
    );
  }
}

export default Hive;