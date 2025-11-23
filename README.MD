# Hive 

![Hive](./img/hive.png)

**Hive** is a lightweight, file-based multimodal document database and vector search engine for Node.js. It's designed for simplicity and efficiency, allowing you to store, retrieve, and search through text and image documents using vector embeddings.

## Features

- **Multimodal Storage**: Store both text and image documents in the same database.
- **Vector Search**: Perform similarity searches using vector embeddings (Text-to-Text, Image-to-Image).
- **Automatic Processing**: Automatically detects file types and generates appropriate embeddings.
- **File Support**: Supports `.txt`, `.doc`, `.docx`, `.pdf` for text, and `.png`, `.jpg`, `.jpeg` for images.
- **Persistence**: Offers both in-memory and on-disk persistence for your data.
- **Configurable Storage**: Store your database anywhere, defaulting to your project root.
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
  // storageDir: "./data", // Optional: Custom storage directory
  pathToDocs: "./documents", // Optional: Auto-load documents from this folder
  logging: true // Optional: Enable processing logs
});

// 2. Add files manually (if not using pathToDocs)
await Hive.addFile("./notes.txt");
await Hive.addFile("./photo.jpg");

// 3. Generate a vector for your query
// For text search:
const textVector = await Hive.embed("your search query", "text");
// For image search:
const imageVector = await Hive.embed("./query_image.jpg", "image");

// 4. Perform a search
const topK = 10; 
const textResults = await Hive.find(textVector, topK);
const imageResults = await Hive.find(imageVector, topK);

// 5. Log the results
console.log("Text Results:", textResults);
console.log("Image Results:", imageResults);
```

When you first initialize Hive with a `pathToDocs`, it will:
1.  Scan the specified directory for supported files (text and images).
2.  Process text files into chunks and images into embeddings.
3.  Create and save a vector database at the specified location.

## API Reference

### `Hive.init(options)`

Initializes the Hive database with the specified configuration.

-   `options` (Object): Configuration options.
    -   `dbName` (String): The name of the database. **Default**: `"Documents"`.
    -   `storageDir` (String): Directory to store the database folder. **Default**: `process.cwd()` (Project Root).
    -   `pathToDB` (String): Full path to the database file (overrides `storageDir`).
    -   `pathToDocs` (String | Boolean): The path to the directory containing your documents. If `false`, no documents will be processed automatically. **Default**: `false`.
    -   `watch` (Boolean): Enable file watching for auto-updates. **Default**: `false`.
    -   `logging` (Boolean): Enable console logging for file processing. **Default**: `false`.
    -   `SliceSize` (Number): Token limit for text slicing. **Default**: `512`.
    -   `minSliceSize` (Number): Minimum token count for a slice to be indexed. **Default**: `100`.
    -   `documents` (Object): An object specifying the file extensions to process.
        -   `text` (Array): **Default**: `[".txt", ".doc", ".docx", ".pdf"]`.
        -   `image` (Array): **Default**: `[".png", ".jpg", ".jpeg"]`.

### `Hive.embed(input, type)`

Generates a vector embedding for the given input.

-   `input` (String): The text content or image file path.
-   `type` (String): The type of embedding to generate (`"text"` or `"image"`). **Default**: `"text"`.

### `Hive.find(queryVector, topK)`

Finds the most similar documents to a given query vector. Automatically filters results to match the dimension of the query vector (e.g., text queries only return text results).

-   `queryVector` (Array): The vector to search with.
-   `topK` (Number): The number of top results to return. **Default**: `10`.

### `Hive.addFile(filePath)`

Adds a single file to the database. Automatically detects if it's text or image based on extension.

-   `filePath` (String): Path to the file.

### `Hive.removeFile(filePath)`

Removes a file and its associated embeddings from the database.

-   `filePath` (String): Path to the file to remove.

### `Hive.insertOne(entry)`

Inserts a single document into the database.

-   `entry` (Object): An object containing the vector and metadata.
    -   `vector` (Array): The vector embedding.
    -   `meta` (Object): Metadata associated with the document.

### `Hive.deleteItem(id)`

Deletes a specific item from the database by its ID.

-   `id` (String): The ID of the item to delete.

### `Hive.updateOne(query, entry)`

Updates an existing document in the database.

-   `query` (Object): A query to find the document to update (e.g., `{ filePath: "/path/to/doc.txt" }`).
    -   `entry` (Object): The new document entry.

### Deprecated

### `Hive.getVector(input, options)`

**Deprecated**: Use `Hive.embed` instead.
Legacy method to generate a vector. Returns an object wrapper around the vector to maintain backward compatibility.

-   `input` (String): The text content or image file path.
-   `options` (Object): Optional parameters.
-   **Returns**: `{ data: Array }`

## Performance

Hive is optimized to handle large datasets. A search in a database with **30,000 entries** takes approximately **30ms** on an AMD Ryzen 7 3700X 8-Core Processor.

## License

This project is licensed under the MIT License.