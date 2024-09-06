const fs = require("fs");
const path = require("path");
const reader = require("any-text");

/**
 * The Hive class is a database management system that provides a simple and efficient way to store and retrieve data.
 * It uses a file-based storage system and supports various operations such as creating collections, inserting data, and querying the database.
 * The class also includes functionality for loading and saving the database to disk, as well as integrating with natural language processing models for feature extraction.
 */
function Hive(){ }

  Hive.init = async function (dbName = "Documents", filePath, pathToDocs = false) {
    Hive.dbName = dbName;
    Hive.filePath = filePath || `./${dbName}/${dbName}.json`; // Default file path for saving/loading
    Hive.collections = new Map();
    Hive.createCollection(dbName);
    Hive.TransOptions = { pooling: "mean", normalize: false };
    Hive.loadToMemory(); // Load to memory automatically
    await Hive.initTransformers();
    if (pathToDocs && fs.existsSync(pathToDocs) && Hive.databaseExists()===false) {
      await Hive.pullDocuments(pathToDocs);
    }
  }
  Hive.databaseExists = function () {
    if (fs.existsSync(Hive.filePath) && fs.statSync(Hive.filePath).size > 200) {
      console.log(`Database exists ${Hive.filePath}`);
      return true;
    }else{
      console.log(`Database does not exist ${Hive.filePath}`);
      return false;
    }
  }

  // Initialize transformers
  Hive.initTransformers = async function (){
    if(!Hive.pipeline){
      const transformersModule = await import("@xenova/transformers");
      Hive.pipeline = transformersModule.pipeline;
      // Define getVector as a function that takes text input and uses the pipeline
      Hive.getVector = await Hive.transInit();
    }else{
      console.log(`Transformers already initialized`);
    }
  }

  Hive.transInit = async function (){
    return await Hive.pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  }

  // Create a collection
  Hive.createCollection = function (dbName) {
    if (!Hive.collections.has(dbName)) {
      Hive.collections.set(dbName, []);
      console.log(`Collection ${dbName} created.`);
    } else {
      console.log(`Collection ${dbName} already exists.`);
    }
  }

  // Insert one object into a specific collection
  Hive.insertOne = function (dbName, entry) {
    if (Hive.collections.has(dbName)) {
      console.log(`Inserting into collection: ${dbName}`);
      const { vector, meta } = entry;
      Hive.collections.get(dbName).push({
        vector: vector,
        meta,
      });
    } else {
      console.log(`Collection ${dbName} does not exist.`);
    }
  }
  // Insert many entries into a collection
  Hive.insertMany = function (dbName, entries) {
    if (Hive.collections.has(dbName)) {
      const collection = Hive.collections.get(dbName);
      for (let i = 0; i < entries.length; i++) {
        const { vector, meta } = entries[i];
        collection.push({
          vector: vector,
          meta,
        });
      }
      Hive.saveToDisk(); // Auto-save after bulk insertion
    } else {
      console.log(`Collection ${dbName} does not exist.`);
    }
  }

  // Save the database to disk
  Hive.saveToDisk= function () {
    const data = {};
    Hive.collections.forEach((value, key) => {
      data[key] = value.map((entry) => ({
        vector: entry.vector, // Convert Float32Array back to Array for JSON
        meta: entry.meta,
      }));
    });

    fs.writeFileSync(Hive.filePath, JSON.stringify(data), "utf8");
    console.log(`Database saved to ${Hive.filePath}`);
  }

  // Load the database into memory from disk
  Hive.loadToMemory= async function (){
    if (fs.existsSync(Hive.filePath)) {
      const rawData = fs.readFileSync(Hive.filePath, "utf8");
      const data = JSON.parse(rawData);
      Hive.collections.clear(); // Clear existing collections
      for (const [dbName, entries] of Object.entries(data)) {
        Hive.createCollection(dbName); // Recreate collections
        entries.forEach((entry) => {
          Hive.collections.get(dbName).push({
            vector: new Float32Array(entry.vector),
            meta: entry.meta,
          });
        });
      }
      console.log(`Database loaded into memory from ${Hive.filePath}`);
    } else {
      console.log(`File ${Hive.filePath} does not exist.`);
    }
  }


  // Find vectors similar to the query vector  
  Hive.find = async function (dbName, queryVector, topK = 5) {
    const queryVectorMag = Hive.normalize(queryVector);
    const results = [];
    if (Hive.collections.has(dbName)) {
      const collection = Hive.collections.get(dbName);
      for (const item of collection) {
        const itemVector = item.vector;
       // console.log("pooled", queryVector.length, "item",itemVector.length);
        const similarity = Hive.cosineSimilarity(queryVector, itemVector, queryVectorMag);
        results.push({ document: item, similarity });
      }
    //   results[0].document.meta
      results.sort((a, b) => b.similarity - a.similarity); // Sort by similarity descending
    } else {
      console.error(`Collection ${dbName} does not exist.`);
    }
    return results.slice(0, topK); // Return top K results
  }
  
  Hive.cosineSimilarity = function (queryVector, itemVector, queryVectorMag) {
    const dotProduct = queryVector.reduce((sum, val, i) => sum + val * itemVector[i], 0);
    const itemVectorMag = Hive.normalize(itemVector);
    return dotProduct / (queryVectorMag * itemVectorMag);
  }
  
  Hive.normalize = function (vector) {
    let sum = 0;
    for (let i = 0; i < vector.length; i++) {
      sum += vector[i] * vector[i];
    }
    return Math.sqrt(sum);
  }
  
  Hive.rank = function (queryVector, results) {
    return results.sort((a, b) => b.distance - a.distance); // Higher similarity (closer to 1) is better
  }

  Hive.tokenCount = function (text) {
    const tokens = text.match(/\b\w+\b/g) || [];
    const tokensarr = tokens.filter((token) => /\S/.test(token));
    // console.log(tokensarr, tokensarr.length);
    return [tokensarr, tokensarr.length];
  }

  Hive.addItem = async function (text, filePath = "") {
    try {
      const vector = await Hive.getVector(text, Hive.TransOptions);
      // Insert the item into the "Documents" collection
      Hive.insertOne("Documents", {
        vector: Array.from(vector.data),
        meta: {
          content: Hive.escapeChars(text),
          href: Hive.escapeChars(filePath),
          title: Hive.escapeChars(text.slice(0, 20)),
        },
      });
    } catch (error) {
      console.error("Error adding item:", error);
    }
  }

  // Read file and tokenize its content, splitting into slices for insertion
  Hive.readFile = async function (filePath, dir) {
    let text = await reader.getText(filePath); // Simulate reading file content
    const [tokens, len] = Hive.tokenCount(text);

    const sliceSize = 512;
    let startIndex = 0;

    while (startIndex < len) {
      let endIndex = startIndex + sliceSize;
      // Ensure we don't split a word
      if (endIndex < len) {
        while (endIndex > startIndex && tokens[endIndex] !== " ") {
          endIndex--;
        }
      }
      if (endIndex === startIndex) {
        endIndex = Math.min(startIndex + sliceSize, len);
      }
      const slice = tokens.slice(startIndex, endIndex);
      await Hive.addItem(slice.join(" "), filePath);
      startIndex = endIndex + 1;
    }
  }

  // Tokenize the text, cleaning it of non-alphanumeric characters
  Hive.tokenize = function (text) {
    const words = text.split(/\s+/);
    return words.filter((word) => word.length > 0 && !word.match(/[^a-zA-Z0-9]/)).join(" ");
  }

  // Pull documents recursively from a directory and process them
  Hive.pullDocuments = async function (dir) {
    const files = await fs.promises.readdir(dir, { withFileTypes: true });
    for (const file of files) {
      const fullPath = path.join(dir, file.name);
      if (file.isDirectory()) {
        await Hive.pullDocuments(fullPath);
      } else if (file.isFile() && [".txt", ".doc", ".docx", ".pdf"].includes(path.extname(file.name))) {
        await Hive.readFile(fullPath, dir);
        console.log(`Processed file: ${fullPath}`);
      }
    }
    Hive.saveToDisk();
  }

  Hive.escapeChars = function (text) {
    // Function to escape special characters in text
    return text.replace(/[&<>"']/g, (match) => {
      const escapeChars = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      };
      return escapeChars[match];
    });
  }

// Hive.init();
module.exports = Hive;
