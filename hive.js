import { pipeline } from "@xenova/transformers";
import fs from "fs";
import path from "path";
import doc2txt from "./doc2txt.js";
import { fileURLToPath } from "url";

/**
 * Copyright Denis Spasyuk
 * The Hive class is a database management system that provides a simple and efficient way to store and retrieve data.
 * It uses a file-based storage system and supports various operations such as creating collections, inserting data, and querying the database.
 * The class also includes functionality for loading and saving the database to disk, as well as integrating with natural language processing models for feature extraction.
 */

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class Hive {
  static collections = new Map();
  static pipelines = {};
  static saveTimeout = null;

  // Default configuration
  static dbName = "Documents";
  static pathToDB = path.join(process.cwd(), "db", "Documents", "Documents.json");
  static pathToDocs = false;
  static type = "text"; // Deprecated but kept for backward compatibility
  static watch = false;
  static logging = false;
  static documents = {
    text: [".txt", ".doc", ".docx", ".pdf"],
    image: [".png", ".jpg", ".jpeg"],
  };
  static SliceSize = 512;
  static minSliceSize = 100;
  static saveInterval = 5000;
  static models = {
    text: "Xenova/all-MiniLM-L6-v2",
    image: "Xenova/clip-vit-base-patch32",
  };
  static TransOptions = { pooling: "mean", normalize: false };
  static escapeRules = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
    "\\": "\\\\",
    "/": "\\/",
  };

  /**
   * Initialize Hive DB with support for both text and image embeddings
   * @param {Object} options
   */
  static async init(options = {}) {
    Hive.dbName = options.dbName || Hive.dbName;
    
    if (options.pathToDB) {
      Hive.pathToDB = options.pathToDB;
    } else if (options.storageDir) {
      Hive.pathToDB = path.join(options.storageDir, Hive.dbName, Hive.dbName + ".json");
    } else {
      // Default to process.cwd() to store in project root, updating with dbName
      Hive.pathToDB = path.join(process.cwd(), "db", Hive.dbName, Hive.dbName + ".json");
    }

    Hive.pathToDocs = options.pathToDocs !== undefined ? options.pathToDocs : Hive.pathToDocs;
    Hive.type = options.type || Hive.type; // Restored for backward compatibility
    Hive.watch = options.watch !== undefined ? options.watch : Hive.watch;
    Hive.logging = options.logging !== undefined ? options.logging : Hive.logging;
    Hive.documents = options.documents || Hive.documents;
    Hive.SliceSize = options.SliceSize || Hive.SliceSize;
    Hive.minSliceSize = options.minSliceSize || Hive.minSliceSize;

    Hive.createCollection(Hive.dbName);
    await Hive.loadToMemory();
    await Hive.initTransformers();

    if (Hive.pathToDocs) {
      if (fs.existsSync(Hive.pathToDocs)) {
        if (!Hive.databaseExists()) {
          await Hive.pullDocuments(Hive.pathToDocs);
        } else {
          // Already loaded to memory above
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

  /**
   * Check if database file exists and has content
   * @returns {boolean}
   */
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
   * Initialize transformers pipelines
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
  }

  /**
   * Legacy support for getVector
   * @param {string} input
   * @param {Object} options
   */
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

  /**
   * Create a collection
   * @param {string} name
   */
  static createCollection(name = Hive.dbName) {
    if (!Hive.collections.has(name)) {
      Hive.collections.set(name, []);
    } else {
      Hive.log(`Collection ${name} already exists.`);
    }
  }

  static randomId() {
    return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
  }

  /**
   * Delete item by ID
   * @param {string} id
   */
  static deleteItem(id) {
    if (Hive.collections.has(Hive.dbName)) {
      const collection = Hive.collections.get(Hive.dbName);
      const filteredCollection = collection.filter((item) => item.id !== id);
      Hive.collections.set(Hive.dbName, filteredCollection);
      Hive.saveToDisk();
    }
  }

  /**
   * Insert one object into a specific collection
   * @param {Object} entry
   */
  static insertOne(entry) {
    if (Hive.collections.has(Hive.dbName)) {
      const { vector, meta } = entry;
      const magnitude = Hive.normalize(vector);
      Hive.log("Inserting entry:", entry);
      Hive.collections.get(Hive.dbName).push({
        vector,
        magnitude,
        meta,
        id: Hive.randomId(),
      });
      Hive.saveToDisk();
    }
  }

  /**
   * Update one entry based on query
   * @param {Object} query
   * @param {Object} entry
   */
  static updateOne(query, entry) {
    if (Hive.collections.has(Hive.dbName)) {
      Hive.findMeta(query, entry);
      Hive.saveToDisk();
    }
  }

  /**
   * Insert many entries into a collection
   * @param {Array} entries
   */
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
    } else {
      Hive.log(`Collection ${Hive.dbName} does not exist.`);
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

  /**
   * Save the database to disk using atomic write
   */
  static saveToDisk() {
    return new Promise((resolve, reject) => {
      if (Hive.saveTimeout) {
        clearTimeout(Hive.saveTimeout);
      }
      Hive.saveTimeout = setTimeout(async () => {
        try {
          const data = {};
          for (const [key, value] of Hive.collections.entries()) {
            const collection = [];
            for (let i = 0; i < value.length; i++) {
              const entry = value[i];
              collection.push({
                vector: Array.from(entry.vector),
                meta: entry.meta,
                magnitude: entry.magnitude,
                id: entry.id,
              });
            }
            data[key] = collection;
          }

          await Hive.ensureDirectoryExists(Hive.pathToDB);
          
          // Atomic write: write to temp file then rename
          const tempPath = `${Hive.pathToDB}.tmp`;
          await fs.promises.writeFile(tempPath, JSON.stringify(data), "utf8");
          await fs.promises.rename(tempPath, Hive.pathToDB);
          
          Hive.log(`Database saved to ${Hive.pathToDB}`);
          resolve();
        } catch (error) {
          console.error("Error saving database:", error);
          reject(error);
        }
      }, Hive.saveInterval);
    });
  }

  /**
   * Load the database into memory from disk
   */
  static async loadToMemory() {
    if (fs.existsSync(Hive.pathToDB) && (Hive.collections.size === 0 || Hive.collections.get(Hive.dbName)?.length === 0)) {
      try {
        const rawData = await fs.promises.readFile(Hive.pathToDB, "utf8");
        const data = JSON.parse(rawData);
        Hive.collections.clear();
        for (const [dbName, entries] of Object.entries(data)) {
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
        Hive.log(`Database loaded into memory from ${Hive.pathToDB}`);
      } catch (error) {
        console.error("Error loading database:", error);
      }
    }
  }

  /**
   * Find and replace metadata in collection
   * @param {Object} query 
   * @param {Object} entry 
   */
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
   * Find similar vectors
   * @param {Array} queryVector 
   * @param {number} topK 
   * @returns {Array}
   */
  static async find(queryVector, topK = 10) {
    const queryVectorMag = Hive.normalize(queryVector);
    const collection = Hive.collections.get(Hive.dbName) || [];
    const results = [];
    
    for (let i = 0; i < collection.length; i++) {
      const item = collection[i];
      // Only compare vectors of the same dimension
      if (item.vector.length === queryVector.length) {
          const similarity = Hive.cosineSimilarity(queryVector, item.vector, queryVectorMag, item.magnitude);
          results.push({ document: item, similarity });
      }
    }
    results.sort((a, b) => b.similarity - a.similarity);
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
    const tokensarr = [];
    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i];
      if (/\S/.test(token)) {
        tokensarr.push(token);
      }
    }
    return [tokensarr, tokensarr.length];
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
      } else {
          throw new Error(`Unsupported type: ${type}`);
      }
      
      const vectorData = result.data || result;
      
      Hive.insertOne({
        vector: Array.isArray(vectorData) ? vectorData : Array.from(vectorData),
        meta: {
          content: type === "text" ? Hive.escapeChars(input) : `Image: ${path.basename(filePath)}`,
          href: filePath,
          title: filename || (type === "text" ? Hive.escapeChars(input.slice(0, 20)) : `Image: ${path.basename(filePath)}`),
          filePath: filePath,
          type: type // Store type in metadata
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
            } catch (error) {
              console.error(`Error reading symlink ${fullPath}:`, error);
              continue;
            }
          }
      
          try {
            const stats = await fs.promises.stat(fullPath);
            if (stats.isDirectory()) {
              await Hive.pullDocuments(fullPath);
            } else if (Hive.documents.text.includes(ext)) {
              await Hive.readFile(fullPath, "text");
            } else if (Hive.documents.image.includes(ext)) {
              await Hive.readFile(fullPath, "image");
            } else {
              // Hive.log(`Skipping unsupported file: ${fullPath}`);
            }
          } catch (error) {
            console.error(`Error processing file ${fullPath}: ${error}`);
          }
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
          Hive.log(`Unsupported file type for update: ${filePath}`);
          return;
      }

      let input = "";
      if (type === "text") {
        const extracted = await doc2txt.extractTextFromFile(filePath);
        input = extracted.text;
      } else {
          input = filePath; // For images, input is the path
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
          Hive.log(`Checking File: ${filePath}`);
          if (!(await Hive.fileExistsInDatabase(filePath))) {
             await Hive.updateFile(filePath);
          }
        })
        .on("change", async (filePath) => {
          Hive.log(`File changed: ${filePath}`);
          await Hive.updateFile(filePath);
        })
        .on("unlink", async (filePath) => {
          Hive.log(`File removed: ${filePath}`);
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
        
        while (startIndex < len) {
          let endIndex = startIndex + Hive.SliceSize;
          endIndex = Math.min(endIndex, len);
          
          const sliceLength = endIndex - startIndex;
          if (sliceLength >= Hive.minSliceSize) {
            const slice = tokens.slice(startIndex, endIndex).join(" ");
            Hive.log(`Slice: ${slice.substring(0, 50)}...`, len, Hive.minSliceSize, startIndex, endIndex);
            await Hive.addItem(slice, filePath, type);
          }
          startIndex = endIndex;
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

  static escapeChars(text) {
    return (
      text
        .replace(/[&<>"'\\\/]/g, (match) => {
          return Hive.escapeRules[match];
        })
        .replace(/\b(?:TEY|FY|AFRL\s+\d+|[0-9]{2,})\b/g, "")
        .replace(/[^A-Za-z0-9\s]/g, "")
        .replace(/\s+/g, " ")
        .replace(/\b([A-Za-z])\b(\s+\1)+/g, "")
        .replace(/\b[A-Za-z]\b/g, "")
        .trim()
    );
  }
}

export default Hive;
