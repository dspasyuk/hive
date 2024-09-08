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

npm install hive 

Usage 


`import Hive from 'hive';`  

`await Hive.init('Documents', './db/Documents/db.json',"./docs");`  


    dbName: Name of the database (default: "Documents") 
    filePath: Path to save/load the database file 
    pathToDocs: Optional path to directory containing documents to process 


First the Hive will take all your text files from ./docs folder, split it in to chunks (512 tokens), 
convert them to chunks and create vector database in './db/Documents/db.json'  

Once database is created you can do a vector search using the following: 

`const vector = await Hive.getVector("Human Factors Workscope", Hive.TransOptions); 
const results = await Hive.find(vector.data, 10); // 10 is topK, number of searches to return  
console.log(results);
`

### Alternative aproach to create the database

By default Hive user transformers.js (Xenova/all-MiniLM-L6-v2) to create vector embeddings for your data and query but you can easily bypass that using inserOne function 

` 
Hive.insertOne({vector: Array,  meta: {}});
`  
### Speed

The database was optimized to handle well above 1,000,000 entries (512 tokens) large.
On Average it takes around 30 ms to search a database with 30,000 entries (AMD Ryzen 7 3700X 8-Core Processor)


License MIT
This project is licensed under the MIT License.
