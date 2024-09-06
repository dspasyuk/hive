const fs = require("fs");
const path = require("path");
const reader = require("any-text");

function Hive(){ }

  Hive.init = async function (dbName, filePath) {
    Hive.dbName = dbName;
    Hive.filePath = filePath || `./${dbName}.json`; // Default file path for saving/loading
    Hive.collections = new Map();
    Hive.TransOptions = { pooling: "mean", normalize: false };
    await Hive.initTransformers();
    Hive.loadToMemory(); // Load to memory automatically
  }

  // Initialize transformers
   Hive.initTransformers = async function (){
    const transformersModule = await import("@xenova/transformers");
    Hive.pipeline = transformersModule.pipeline;
    // Define getVector as a function that takes text input and uses the pipeline
    Hive.getVector = await Hive.transInit();
  }

  Hive.transInit = async function (){
    return await Hive.pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  }

  // Create a collection
  Hive.createCollection = function (collectionName) {
    if (!Hive.collections.has(collectionName)) {
      Hive.collections.set(collectionName, []);
      console.log(`Collection ${collectionName} created.`);
    } else {
      console.log(`Collection ${collectionName} already exists.`);
    }
  }

  // Insert one object into a specific collection
  Hive.insertOne = function (collectionName, entry) {
    if (Hive.collections.has(collectionName)) {
      console.log(`Inserting into collection: ${collectionName}`);
      const { vector, meta } = entry;
      Hive.collections.get(collectionName).push({
        vector: vector,
        meta,
      });
    } else {
      console.log(`Collection ${collectionName} does not exist.`);
    }
  }
  // Insert many entries into a collection
  Hive.insertMany = function (collectionName, entries) {
    if (Hive.collections.has(collectionName)) {
      const collection = Hive.collections.get(collectionName);
      for (let i = 0; i < entries.length; i++) {
        const { vector, meta } = entries[i];
        collection.push({
          vector: new Float32Array(vector),
          meta,
        });
      }
      Hive.saveToDisk(); // Auto-save after bulk insertion
    } else {
      console.log(`Collection ${collectionName} does not exist.`);
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
  Hive.loadToMemory= function (){
    if (fs.existsSync(Hive.filePath)) {
      const rawData = fs.readFileSync(Hive.filePath, "utf8");
      const data = JSON.parse(rawData);

      Hive.collections.clear(); // Clear existing collections

      for (const [collectionName, entries] of Object.entries(data)) {
        Hive.createCollection(collectionName); // Recreate collections
        entries.forEach((entry) => {
          Hive.collections.get(collectionName).push({
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
  // Normalize a vector
  

  // Find vectors similar to the query vector
  Hive.meanPooling = function (vector, targetLength) {
    const pooledVector = [];
    const poolSize = Math.ceil(vector.length / targetLength);
    for (let i = 0; i < targetLength; i++) {
      const start = i * poolSize;
      const end = Math.min(start + poolSize, vector.length);
      const sum = vector.slice(start, end).reduce((a, b) => a + b, 0);
      pooledVector.push(sum / (end - start));
    }
    return pooledVector;
  }
  
  Hive.find = function (collectionName, queryVector, topK = 5) {
    const results = [];
    if (Hive.collections.has(collectionName)) {
      const collection = Hive.collections.get(collectionName);
  
      for (const item of collection) {
        const itemVector = item.vector;
        const itemLength = itemVector.length;
       // console.log("pooled", queryVector.length, "item",itemVector.length);
        const similarity = Hive.cosineSimilarity(queryVector, itemVector);
        results.push({ document: item, similarity });
      }
    //   results[0].document.meta
      results.sort((a, b) => b.similarity - a.similarity); // Sort by similarity descending
     
    } else {
      console.error(`Collection ${collectionName} does not exist.`);
    }
    return results.slice(0, topK); // Return top K results
  }
  
  Hive.cosineSimilarity = function (vector1, vector2) {
    const dotProduct = vector1.reduce((sum, val, i) => sum + val * vector2[i], 0);
    const magnitudeA = Hive.normalize(vector1);
    const magnitudeB = Hive.normalize(vector2);
    return dotProduct / (magnitudeA * magnitudeB);
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
// rank(queryVector, results) {
//     const tfidfScores = results.map((result) => {
//       const tf = calculateTermFrequency(result.document.meta.content, queryVector);
//       const idf = calculateInverseDocumentFrequency(queryVector, results);
//       return tf * idf;
//     });
  
//     return results.sort((a, b) => tfidfScores[results.indexOf(b)] - tfidfScores[results.indexOf(a)]);
//   }
  
//   calculateTermFrequency(text, query) {
//     const terms = text.split(" ");
//     const queryTerms = query;
//     const tf = queryTerms.reduce((acc, term) => {
//       const frequency = terms.filter((t) => t === term).length;
//       return acc + frequency;
//     }, 0);
//     return tf / terms.length;
//   }
  
//   calculateInverseDocumentFrequency(query, results) {
//     const numDocuments = results.length;
//     const numDocumentsWithTerm = results.filter((result) => {
//       return result.document.meta.content.includes(query);
//     }).length;
//     return Math.log(numDocuments / numDocumentsWithTerm);
//   }



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
      await Hive.addItem(slice.join(" "), path.relative(dir, filePath));
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
