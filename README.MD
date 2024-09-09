# Hive (Work in progress)

![Hive](./img/hive.png)

Hive is a lightweight document database and vector search engine built with Node.js. It provides an efficient way to store, retrieve, and search documents using vector embeddings.

### Features

Document storage and retrieval   
Vector-based similarity search  
Automatic document processing and tokenization  
Support for various file formats (txt, doc, docx, pdf) (any-text)  
In-memory and on-disk persistence  

### Installation 
```
npm install deonis/hive 
```

### Usage 
```import Hive from 'hive';  
await Hive.init(dbName, filePath, pathToDocs);
```

dbName: Name of the database (default: "Documents") 
filePath: Path to save/load the database file 
pathToDocs: Optional path to directory containing documents to process 
```
await Hive.init('Documents', './db/Documents/db.json',"./docs");
```


First the Hive will take all your text files from ./docs folder, split it in to chunks (512 tokens), 
convert them  vectors and create vector database in './db/Documents/db.json'  

Once database is created you can do a vector search using the following: 

```
const vector = await Hive.getVector("Hello World", Hive.TransOptions); 
const results = await Hive.find(vector.data, 10); // 10 is topK, number of searches to return  
console.log(results);
```

### Alternative aproach to create the database

By default Hive user transformers.js (Xenova/all-MiniLM-L6-v2) to create vector embeddings for your data and query but you can easily bypass that using inserOne function 

```
Hive.insertOne({vector: Array,  meta: {}});
```
### Speed

The database was optimized to handle well above 1,000,000 entries (512 tokens) large.
On Average it takes around 30 ms to search a database with 30,000 entries (AMD Ryzen 7 3700X 8-Core Processor)

### Hive Database Class

```Hive.insertOne(entry)```

    entry: An object containing the vector and metadata for the document.

Inserts a single entry into the collection. The entry includes the vector (features extracted from text) and associated metadata.


```Hive.insertMany(entries)```

    entries: An array of objects, where each object contains a vector and meta.

Inserts multiple entries into the collection at once. Automatically saves the database to disk after bulk insertion.

### Query Data

```Hive.find(queryVector, topK = 5)```

    queryVector: The vector to compare against.
    topK: Number of top similar results to return (default: 5).

Finds the top K vectors similar to the queryVector based on cosine similarity. Returns the most similar documents from the collection.

### Save & Load

```Hive.saveToDisk()```

Saves the current state of the database to a file on disk, ensuring the data persists across sessions.

```Hive.loadToMemory()```

Loads the database from disk into memory. It parses the data and restores the collection with the precomputed vectors and metadata.

Transformers Integration

Hive.initTransformers()

Initializes the transformers model for feature extraction. This uses a pre-trained model (Xenova/all-MiniLM-L6-v2) to generate vectors for the text content.
Document Processing

Hive.addItem(text, filePath = "")

    text: The textual content to be processed.
    filePath: Optional file path metadata.

Extracts features (vector) from the given text and stores it in the database along with metadata, such as the file path and content snippet.

Hive.readFile(filePath, dir)

    filePath: Path to the document.
    dir: The directory containing the document.

Reads the content of a file, tokenizes the text, and inserts it into the collection after splitting it into manageable slices.

Hive.pullDocuments(dir)

    dir: The directory containing documents.

Recursively reads all documents from the specified directory, processes them, and adds their content to the collection. Supports file types such as .txt, .doc, .docx, and .pdf.
Utility Functions

Hive.tokenCount(text)

    text: Input text to be tokenized.

Tokenizes the given text and returns the tokens and token count.

Hive.normalize(vector)

    vector: The vector to normalize.

Computes the magnitude of the vector to facilitate cosine similarity calculations.

Hive.cosineSimilarity(queryVector, itemVector, queryVectorMag, itemVectorMag)

    queryVector: The vector of the query document.
    itemVector: The vector of a document in the collection.
    queryVectorMag: The magnitude of the query vector.
    itemVectorMag: The magnitude of the item vector.

Calculates the cosine similarity between two vectors.


Hive.escapeChars(text)

Cleans text by escaping special characters, removing repeated single-letter words, and trimming unnecessary whitespace.


License MIT
This project is licensed under the MIT License.
