# Hive 

![Hive](./img/hive.png)

**Hive** is a lightweight, file-based multimodal document database and vector search engine for Node.js. It's designed for simplicity and efficiency, allowing you to store, retrieve, and search through text and image documents using vector embeddings.

## Features

- **Multimodal Storage**: Store both text and image documents in the same database.
- **Vector Search**: Perform similarity searches using vector embeddings (Text-to-Text, Image-to-Image).
- **Automatic Processing**: Automatically detects file types and generates appropriate embeddings.
- **File Support**: Supports `.txt`, `.doc`, `.docx`, `.pdf` for text, and `.png`, `.jpg`, `.jpeg` for images.
- **Persistence**: Offers both in-memory and heavily optimized on-disk binary `.bin` persistence containing serialized `msgpackr` Float32 arrays for blazing-fast database initialization.
- **Configurable Storage**: Store your database anywhere, defaulting to your project root.
- **Customizable**: Easily extendable to support other file types and embedding models.
- **Authentication & Access Control**: MongoDB-like role-based access control with user management, password hashing, and configurable permissions per database.

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

## Authentication & Access Control

Hive supports MongoDB-like role-based access control. Configure users and their permissions to restrict database operations.

### Built-in Roles

| Role | Actions | Description |
|------|---------|-------------|
| `root` | All actions (`*`) | Full access to everything |
| `dbOwner` | All actions (`*`) | Full access to the database |
| `readWrite` | `find`, `insert`, `update`, `delete`, `embed` | Read and write data |
| `read` | `find`, `embed` | Read-only access |
| `dbAdmin` | `admin` | User management and database administration |

### Quick Start with Auth

```javascript
import Hive from '@deonis/hive';

// Initialize with users (bootstrap)
await Hive.init({
  dbName: "MyDB",
  permissions: {
    users: {
      admin: { password: "secret", roles: ["dbOwner"] },
      viewer: { password: "view123", roles: ["read"] }
    },
    autoAuth: "admin" // Auto-authenticate after init
  }
});

// Or create users programmatically (allowed when no users exist)
Hive.createUser({ user: "admin", pwd: "secret", roles: ["dbOwner"] });

// Authenticate
Hive.auth("admin", "secret");
Hive.whoAmI(); // "admin"

// Operations check permissions automatically
await Hive.find(query, 10); // ✓ allowed for all roles
Hive.insertOne(entry);      // ✓ allowed for admin/readWrite/dbOwner/root

// Login as a read-only user
Hive.auth("viewer", "view123");
Hive.insertOne(entry);      // ✗ throws: not authorized
```

### Auth Behavior

- **No users configured**: All operations proceed without authentication (backward compatible).
- **Users configured**: Every database operation checks that the current user has the required role/action.
- **Bootstrap**: When no users exist, `createUser()` is allowed without authentication so you can create your first admin.
- **Persistence**: Users and roles are stored in the `.bin` database file and restored on reload.

## API Reference

### `Hive.init(options)`

Initializes the Hive database with the specified configuration.

-   `options` (Object): Configuration options.
    -   `dbName` (String): The name of the database. **Default**: `"Documents"`.
    -   `storageDir` (String): Directory to store the database folder. **Default**: `process.cwd()` (Project Root).
    -   `pathToDB` (String): Full path to the database file, should ideally end in `.bin` (overrides `storageDir`). Note: Legacy `.json` files will be automatically migrated to `.bin` format upon loading.
    -   `pathToDocs` (String | Boolean): The path to the directory containing your documents. If `false`, no documents will be processed automatically. **Default**: `false`.
    -   `watch` (Boolean): Enable file watching for auto-updates. **Default**: `false`.
    -   `logging` (Boolean): Enable console logging for file processing. **Default**: `false`.
    -   `SliceSize` (Number): Token limit for text slicing. **Default**: `512`.
    -   `minSliceSize` (Number): Minimum token count for a slice to be indexed. **Default**: `100`.
    -   `overlap` (Number): Overlap between chunks. Can be a percentage (< 1) or token count (>= 1). **Default**: `5%` of `SliceSize`.
    -   `documents` (Object): An object specifying the file extensions to process.
        -   `text` (Array): **Default**: `[".txt", ".doc", ".docx", ".pdf"]`.
        -   `image` (Array): **Default**: `[".png", ".jpg", ".jpeg"]`.
    -   `permissions` (Object): Authentication configuration.
        -   `users` (Object): A map of username to user config.
            -   `password` (String): Plain-text password (hashed with SHA-256 + salt on init).
            -   `roles` (Array): Roles to assign (e.g., `["dbOwner"]`, `["read"]`).
        -   `autoAuth` (String): Username to auto-authenticate after init.

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

### `Hive.deleteOne(id)`

Deletes a specific item from the database by its ID.

-   `id` (String): The ID of the item to delete.

### `Hive.updateOne(query, entry)`

Updates an existing document in the database.

-   `query` (Object): A query to find the document to update (e.g., `{ filePath: "/path/to/doc.txt" }`).
    -   `entry` (Object): The new document entry.

### `Hive.auth(username, password)`

Authenticates a user for the current session.

-   `username` (String): The username.
-   `password` (String): The password.
-   **Throws**: If the user doesn't exist or password is incorrect.

### `Hive.logout()`

Clears the current authenticated user.

### `Hive.whoAmI()`

Returns the username of the currently authenticated user, or `null`.

-   **Returns**: `String | null`

### `Hive.createUser({ user, pwd, roles })`

Creates a new user. Requires `admin` privilege unless no users exist (bootstrap).

-   `user` (String): Username.
-   `pwd` (String): Password.
-   `roles` (Array): Roles to assign. **Default**: `["read"]`.

### `Hive.dropUser(username)`

Deletes a user. Requires `admin` privilege.

-   `username` (String): Username to delete.
-   **Throws**: If trying to delete yourself.

### `Hive.getUsers()`

Lists all users with their roles. Requires `admin` privilege.

-   **Returns**: `Array<{ user: String, roles: Array }>`

### Deprecated

### `Hive.getVector(input, options)`

**Deprecated**: Use `Hive.embed` instead.
Legacy method to generate a vector. Returns an object wrapper around the vector to maintain backward compatibility.

-   `input` (String): The text content or image file path.
-   `options` (Object): Optional parameters.
-   **Returns**: `{ data: Array }`

## Python Support

Hive can now be used in Python applications using the `hive.py` wrapper!

### Prerequisites

-   Ensure Node.js is installed and in your system PATH.
-   Install Hive dependencies: `npm install` (in the hive directory).

### Usage

```python
from hive import Hive

# Initialize
with Hive() as hive:
    hive.init({"dbName": "MyDocuments"})

    # Add a file
    hive.add_file("./notes.txt")

    # Search
    vector = hive.embed("search query")
    results = hive.find(vector)
    
    print(results)
```

## Performance

Hive is optimized to handle large datasets. A search in a database with **30,000 entries** takes approximately **30ms** on an AMD Ryzen 7 3700X 8-Core Processor.

## License

This project is licensed under the MIT License.
