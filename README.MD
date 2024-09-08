# Hive 

![Hive](./hive.png)

Hive is a lightweight document database and vector search engine built with Node.js. It provides an efficient way to store, retrieve, and search documents using vector embeddings.
Features

Document storage and retrieval
Vector-based similarity search
Automatic document processing and tokenization
Support for various file formats (txt, doc, docx, pdf)
In-memory and on-disk persistence

### Installation 

npm install hive

Usage

javascript
const Hive = require('hive-db');

### Initialize Hive 

await Hive.init('MyDatabase', './mydb.json', './documents');


### API 

Hive.init(dbName, filePath, pathToDocs)
Initializes the Hive database.

    dbName: Name of the database (default: "Documents")
    filePath: Path to save/load the database file
    pathToDocs: Optional path to directory containing documents to process

Hive.addItem(text, filePath)
Adds a single item to the database.
Hive.find(dbName, queryVector, topK)
Finds similar vectors in the specified collection.
Hive.pullDocuments(dir)
Recursively processes and adds documents from a directory.
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
License
This project is licensed under the MIT License.
