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
await Hive.init('Documents', './db/Documents/db.json', "./docs");
```


First the Hive will take all your text files from ./docs folder, split it in to chunks (512 tokens), 
convert them  vectors and create vector database in './db/Documents/db.json'  

Once database is created you can do a vector search using the following: 

```
const vector = await Hive.getVector("Hello World", Hive.TransOptions); // uses transformer.js to generate a vector
const results = await Hive.find(vector.data, 10); // 10 is topK, number of searches to return  
console.log(results);
```

### Options Description

    Hive.sliceSize (number): Defines the size of text slices for large documents during vector generation. The default value is 512.  

    Hive.models (object): Specifies the models used for generating embeddings.  
        text (string): The model for text embeddings, e.g., "Xenova/all-MiniLM-L6-v2".  
        image (string): The model for image embeddings, e.g., "Xenova/clip-vit-base-patch16".  

    Hive.documents (object): Defines the types of documents that can be processed for embeddings.  
        text (array): List of supported text file extensions, e.g., [".txt", ".doc", ".docx", ".pdf"].  
        image (array): List of supported image file extensions, e.g., [".png", ".jpg", ".jpeg"].  

    Hive.transformersOptions (object): Configures options for transformers.  
        pooling (string): Specifies the pooling method, e.g., "mean".  
        normalize (boolean): Determines if vectors should be normalized. Default is false.  

    Hive.escapeRules (object): Defines escape sequences for special characters during text parsing.  
        &: &amp;  
        <: &lt;  
        >: &gt;  
        ": &quot;  
        ': &#39;  
        \: \\  
        /: \/  

These options allow you to customize how Hive processes and stores text and image embeddings. Adjust these settings based on your specific requirements and the types of documents you plan to work with.

### Alternative aproach of creating a database

By default Hive user transformers.js (Xenova/all-MiniLM-L6-v2) to create vector embeddings for your data and query but you can easily bypass that using inserOne function 

```
Hive.insertOne({vector: Array,  meta: {}});
```
### Speed

The database was optimized to handle well above 1,000,000 entries (512 tokens) large.
On Average it takes around 30 ms to search a database with 30,000 entries (AMD Ryzen 7 3700X 8-Core Processor)

### Insert Data

```Hive.insertOne(entry)```

    entry: An object containing the vector and metadata for the document.

Inserts a single entry into the collection. The entry includes the vector (features extracted from text) and associated metadata.

### Insert Array of Data

```Hive.insertMany(entries)```

    entries: An array of objects, where each object contains a vector and meta.

Inserts multiple entries into the collection at once. Automatically saves the database to disk after bulk insertion.

### Query Data

```Hive.find(queryVector, topK = 5)```

    queryVector: The vector to compare against.
    topK: Number of top similar results to return (default: 5).

Finds the top K vectors similar to the queryVector based on cosine similarity. Returns the most similar documents from the collection.


License MIT
This project is licensed under the MIT License.
