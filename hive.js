#!/usr/bin/env node
import { pipeline } from "@xenova/transformers";
import fs from "fs";
import path from "path";
import doc2txt from "./doc2txt.js";
import { fileURLToPath } from "url";
import readline from "readline";

/**
 * Copyright Denis Spasyuk
 * The Hive class is a database management system that provides a simple and efficient way to store and retrieve data.
 * Updated to support Hybrid Search (Cosine + Reranking) and Scientific Data.
 */

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class Hive {
  static collections = new Map();
  static pipelines = {};
  static saveTimeout = null;
  static isSaving = false;

  // Default configuration
  static dbName = "Documents";
  static pathToDB = path.join(process.cwd(), "db", "Documents", "Documents.json");
  static pathToDocs = false;
  static type = "text"; 
  static watch = false;
  static logging = false;
  static overlap = 0; // Default overlap
  // NEW: Reranking Configuration
  static useRerank = false;


  static documents = {
    text: [".txt", ".doc", ".docx", ".pdf"],
    image: [".png", ".jpg", ".jpeg"],
  };
  static SliceSize = 512;
  static minSliceSize = 100;
  static saveInterval = 5000;
  
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
      Hive.pathToDB = options.pathToDB;
    } else if (options.storageDir) {
      Hive.pathToDB = path.join(options.storageDir, Hive.dbName, Hive.dbName + ".json");
    } else {
      Hive.pathToDB = path.join(process.cwd(), "db", Hive.dbName, Hive.dbName + ".json");
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

  static randomId() {
    return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
  }

  static deleteOne(id) {
    if (Hive.collections.has(Hive.dbName)) {
      const collection = Hive.collections.get(Hive.dbName);
      const filteredCollection = collection.filter((item) => item.id !== id);
      Hive.collections.set(Hive.dbName, filteredCollection);
      Hive.saveToDisk();
    }
  }

  static insertOne(entry) {
    if (Hive.collections.has(Hive.dbName)) {
      const { vector, meta } = entry;
      const magnitude = Hive.normalize(vector);
      Hive.log("Inserting entry:", entry.meta.title);
      Hive.collections.get(Hive.dbName).push({
        vector,
        magnitude,
        meta,
        id: Hive.randomId(),
      });
      Hive.saveToDisk();
    }
  }

  static updateOne(query, entry) {
    if (Hive.collections.has(Hive.dbName)) {
      Hive.findMeta(query, entry);
      Hive.saveToDisk();
    }
  }

  static insertMany(entries) {
    if (Hive.collections.has(Hive.dbName)) {
      const collection = Hive.collections.get(Hive.dbName);
      for (let i = 0; i < entries.length; i++) {
        const { vector, meta } = entries[i];
        collection.push({
          vector: vector,
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
    return new Promise((resolve, reject) => {
      if (Hive.saveTimeout) {
        clearTimeout(Hive.saveTimeout);
      }
      Hive.saveTimeout = setTimeout(async () => {
        if (Hive.isSaving) {
            // If already saving, schedule another save after a short delay
            Hive.saveToDisk(); 
            return;
        }
        
        Hive.isSaving = true;
        try {
          await Hive.ensureDirectoryExists(Hive.pathToDB);
          const tempPath = `${Hive.pathToDB}.tmp`;
          const fileStream = fs.createWriteStream(tempPath, { flags: 'w' });
          
          let error = null;
          fileStream.on('error', (err) => { error = err; });

          for (const [key, value] of Hive.collections.entries()) {
            for (const entry of value) {
                const record = {
                    collection: key,
                    id: entry.id,
                    vector: Array.from(entry.vector),
                    meta: entry.meta,
                    magnitude: entry.magnitude
                };
                const line = JSON.stringify(record) + '\n';
                if (!fileStream.write(line)) {
                    await new Promise(resolve => fileStream.once('drain', resolve));
                }
            }
          }
          
          fileStream.end();
          await new Promise((resolve, reject) => {
              fileStream.on('finish', resolve);
              fileStream.on('error', reject);
          });

          if (error) throw error;

          await fs.promises.rename(tempPath, Hive.pathToDB);
          
          Hive.log(`Database saved to ${Hive.pathToDB}`);
          resolve();
        } catch (error) {
          console.error("Error saving database:", error);
          reject(error);
        } finally {
            Hive.isSaving = false;
        }
      }, Hive.saveInterval);
    });
  }

  static async loadToMemory() {
    if (fs.existsSync(Hive.pathToDB) && (Hive.collections.size === 0 || Hive.collections.get(Hive.dbName)?.length === 0)) {
      try {
        Hive.collections.clear();
        const fileStream = fs.createReadStream(Hive.pathToDB);
        const rl = readline.createInterface({
            input: fileStream,
            crlfDelay: Infinity
        });

        for await (const line of rl) {
            if (!line.trim()) continue;
            try {
                // Try parsing as NDJSON record
                const record = JSON.parse(line);
                
                // Check if it's the legacy format (whole file is one JSON object)
                // Legacy format: { "Documents": [...] }
                if (record && !record.collection && !record.vector && typeof record === 'object') {
                     // It's likely the legacy format loaded as a single line (if minified) 
                     // or we need to handle it differently. 
                     // But since we are rebuilding, let's assume NDJSON or try to extract.
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

                // NDJSON format
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
        Hive.log(`Database loaded into memory from ${Hive.pathToDB}`);
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
   * Find similar vectors with optional Re-ranking
   * @param {Array|string} queryInput - Vector or text
   * @param {number} topK 
   * @returns {Array}
   */
  static async find(queryInput, topK = 10) {
    let queryVector = queryInput;
    let queryText = "";
    
    // Auto-detect if input is text string, store it for re-ranking
    if (typeof queryInput === "string") {
        queryText = queryInput;
        queryVector = await Hive.embed(queryInput, "text");
    }

    const queryVectorMag = Hive.normalize(queryVector);
    const collection = Hive.collections.get(Hive.dbName) || [];
    let results = [];
    
    // 1. Initial Retrieval (Cosine Similarity)
    for (let i = 0; i < collection.length; i++) {
      const item = collection[i];
      if (item.vector.length === queryVector.length) {
          const similarity = Hive.cosineSimilarity(queryVector, item.vector, queryVectorMag, item.magnitude);
          results.push({ document: item, similarity });
      }
    }
    
    // Sort by Cosine Similarity
    results.sort((a, b) => b.similarity - a.similarity);

    // 2. Re-ranking (Cross Encoder)
    if (Hive.useRerank && Hive.pipelines.reranker && queryText) {
        // Fetch a larger candidate pool (3x topK) to rerank
        const candidates = results.slice(0, topK * 3);
        const rerankedResults = [];

        for (const res of candidates) {
            // Truncate doc text to ~1000 chars to fit context window
            const docText = res.document.meta.content ? res.document.meta.content.slice(0, 1000) : "";
            
            try {
                // Cross-encoder scores the pair [Query, Document]
                const output = await Hive.pipelines.reranker(queryText, { text_pair: docText });
                
                // Extract score (handling different pipeline output formats)
                const score = output[0]?.score || output.score || 0;
                
                rerankedResults.push({
                    document: res.document,
                    similarity: score, // This is now a relevance score, not cosine
                    original_similarity: res.similarity 
                });
            } catch (err) {
                // Keep original if rerank fails
                rerankedResults.push(res);
            }
        }
        
        // Sort by Cross-Encoder score
        rerankedResults.sort((a, b) => b.similarity - a.similarity);
        return rerankedResults.slice(0, topK);
    }

    return results.slice(0, topK);
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

  static tokenCount(text) {
    const tokens = text.match(/\b\w+\b/g) || [];
    return [tokens, tokens.length];
  }

  static async addFile(filePath, filename) {
    try {
      const ext = path.extname(filePath).toLowerCase();
      if (Hive.documents.text.includes(ext)) {
          const { text } = await doc2txt.extractTextFromFile(filePath);
          if (typeof text === 'string' && text.trim().length > 0) {
            await Hive.addItem(text, filePath, "text", filename);
          }
      } else if (Hive.documents.image.includes(ext)) {
          await Hive.addItem("", filePath, "image", filename);
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
        vector: Array.isArray(vectorData) ? vectorData : Array.from(vectorData),
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
            } else if (Hive.documents.text.includes(ext)) {
              await Hive.readFile(fullPath, "text");
            } else if (Hive.documents.image.includes(ext)) {
              await Hive.readFile(fullPath, "image");
            }
          } catch (error) { }
        }
        Hive.saveToDisk();
    } catch (err) {
        console.error(`Error reading directory ${dir}:`, err);
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
        const extracted = await doc2txt.extractTextFromFile(filePath);
        input = extracted.text;
      } else {
          input = filePath; 
      }

      let result;
      if (type === "text") {
          result = await Hive.pipelines.text(input, Hive.TransOptions);
      } else {
          result = await Hive.pipelines.image(input, Hive.TransOptions);
      }

      const vectorData = result.data || result;
      
      const newEntry = { 
        vector: Array.isArray(vectorData) ? vectorData : Array.from(vectorData), 
        meta: { 
            filePath, 
            type,
            content: type === "text" ? Hive.escapeChars(input) : `Image: ${path.basename(filePath)}`,
            title: type === "text" ? Hive.escapeChars(input.slice(0, 20)) : `Image: ${path.basename(filePath)}`,
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
    const collection = Hive.collections.get(Hive.dbName);
    if (!collection) return false;
    return collection.some((item) => item.meta.filePath === filePath);
  }

  static async readFile(filePath, type) {
    try {
      if (type === "text") {
        let { text } = await doc2txt.extractTextFromFile(filePath);
        const [tokens, len] = Hive.tokenCount(text);
          let startIndex = 0;
          
          // Calculate overlap in tokens
          let overlapTokens = 0;
          if (Hive.overlap < 1 && Hive.overlap > 0) {
             // Percentage
             overlapTokens = Math.floor(Hive.SliceSize * Hive.overlap);
          } else {
             // Absolute token count
             overlapTokens = Math.floor(Hive.overlap);
          }

          // Ensure overlap is not greater than SliceSize (prevent infinite loop)
          if (overlapTokens >= Hive.SliceSize) {
             overlapTokens = Hive.SliceSize - 1; 
          }

          while (startIndex < len) {
            let endIndex = startIndex + Hive.SliceSize;
            endIndex = Math.min(endIndex, len);

            const sliceLength = endIndex - startIndex;
            if (sliceLength >= Hive.minSliceSize) {
              const slice = tokens.slice(startIndex, endIndex).join(" ");
              await Hive.addItem(slice, filePath, type);
            }
            
            // Move start index by stride (SliceSize - overlap)
            // If we reached the end, break
            if (endIndex === len) break;
            
            startIndex += (Hive.SliceSize - overlapTokens);
          }
      } else if (type === "image") {
        await Hive.addItem("", filePath, type);
      }
    } catch (error) {
      console.error(`Error reading file ${filePath}:`, error);
    }
  }


  /**
   * Embed input using the specified type
   * @param {string} input - Text content or image path
   * @param {string} type - "text" or "image"
   * @returns {Promise<Array>}
   */
  static async embed(input, type = "text") {
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