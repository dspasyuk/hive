# Hive 

![Hive](./img/hive.png)

**Hive** is a lightweight, file-based document database and vector search engine for Node.js. It's designed for simplicity and efficiency, allowing you to store, retrieve, and search through documents using vector embeddings.

## Features

- **Document Storage**: Store documents with associated metadata.
- **Vector Search**: Perform similarity searches using vector embeddings.
- **Automatic Processing**: Automatically process and tokenize various file formats.
- **File Support**: Supports `.txt`, `.doc`, `.docx`, and `.pdf` out of the box.
- **Persistence**: Offers both in-memory and on-disk persistence for your data.
- **Customizable**: Easily extendable to support other file types and embedding models.

## Installation

You can install Hive using npm:

```bash
npm i @deonis/hive
```

Alternatively, you can clone the repository directly:

```bash
git clone https://github.com/dspasyuk/hive
```

## Quick Start

Here's a basic example of how to initialize Hive and perform a vector search:

```javascript
import Hive from '@deonis/hive';

// 1. Initialize Hive
await Hive.init({
  dbName: "MyDocuments",
  pathToDB: "/path/to/your/database.json",
  pathToDocs: "/path/to/your/documents",
});

// 2. Generate a vector for your query
const queryVector = await Hive.getVector("your search query");

// 3. Perform a search
const topK = 10; // Number of results to return
const results = await Hive.find(queryVector.data, topK);

// 4. Log the results
console.log(results);
```

When you first initialize Hive with a `pathToDocs`, it will:
1.  Scan the specified directory for supported text files.
2.  Split the files into smaller chunks.
3.  Convert each chunk into a vector embedding.
4.  Create and save a vector database at the specified `pathToDB`.

## API Reference

### `Hive.init(options)`

Initializes the Hive database with the specified configuration.

-   `options` (Object): Configuration options.
    -   `dbName` (String): The name of the database. **Default**: `"Documents"`.
    -   `pathToDB` (String): The full path to the database file, including the filename. **Default**: `db/Documents/Documents.json` inside the module directory.
    -   `pathToDocs` (String | Boolean): The path to the directory containing your documents. If `false`, no documents will be processed automatically. **Default**: `false`.
    -   `documents` (Object): An object specifying the file extensions to process for different data types.
        -   `text` (Array): An array of file extensions for text documents. **Default**: `[".txt", ".doc", ".docx", ".pdf"]`.
    -   `minSliceSize` (Number): The minimum size (in tokens) for a text chunk to be processed. **Default**: `200`.

### `Hive.getVector(text, options)`

Generates a vector embedding for the given text.

-   `text` (String): The input text to convert into a vector.
-   `options` (Object): Transformer options. **Default**: `{ pooling: "mean", normalize: false }`.

### `Hive.find(queryVector, topK)`

Finds the most similar documents to a given query vector.

-   `queryVector` (Array): The vector to search with.
-   `topK` (Number): The number of top results to return. **Default**: `10`.

### `Hive.insertOne(entry)`

Inserts a single document into the database. This is useful for manually adding data without a source file.

-   `entry` (Object): An object containing the vector and metadata.
    -   `vector` (Array): The vector embedding.
    -   `meta` (Object): Metadata associated with the document.

**Example:**

```javascript
Hive.insertOne({
  vector: [0.1, 0.2, ...],
  meta: { content: "This is a manually added document." }
});
```

### `Hive.updateOne(query, entry)`

Updates an existing document in the database.

-   `query` (Object): A query to find the document to update (e.g., `{ filePath: "/path/to/doc.txt" }`).
-   `entry` (Object): The new document entry.

### `Hive.insertMany(entries)`

Inserts an array of documents into the database in a single operation.

-   `entries` (Array): An array of document objects to insert.

## Performance

Hive is optimized to handle large datasets. A search in a database with **30,000 entries** takes approximately **30ms** on an AMD Ryzen 7 3700X 8-Core Processor.

## License

This project is licensed under the MIT License.