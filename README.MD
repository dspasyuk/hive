# Hive

![Hive](./img/hive.png)

**Hive** is a lightweight, file-based multimodal vector database and semantic search engine for Node.js. It stores, retrieves, and searches text and image documents using vector embeddings — no external services required.

## Features

- **Multimodal** — Store text and images in the same database
- **Semantic Search** — Text-to-text and image-to-image similarity search
- **Auto-processing** — Automatically detects file types and generates embeddings
- **File support** — `.txt`, `.doc`, `.docx`, `.pdf` (text); `.png`, `.jpg`, `.jpeg` (images)
- **Binary persistence** — On-disk `.bin` with `msgpackr`-serialized `Float32Array` for fast startup
- **Configurable chunking** — Configurable `SliceSize`, `overlap`, and `minSliceSize`
- **File watching** — Watch directories for changes with `chokidar`
- **Cross-encoder reranking** — Optional reranking pass for improved search quality
- **Authentication & RBAC** — MongoDB-like role-based access control with hashed passwords
- **MCP Server** — Expose Hive as tools for AI agents via the Model Context Protocol
- **Python wrapper** — Use Hive from Python via `hive.py`
- **Custom models** — Plug in any `@xenova/transformers` compatible model

## Installation

```bash
npm i @deonis/hive
```

Or clone the repo:

```bash
git clone https://github.com/dspasyuk/hive.git
```

## Quick Start

```javascript
import Hive from '@deonis/hive';

await Hive.init({
  dbName: "MyDocuments",
  pathToDocs: "./documents",
  logging: true
});

// Add files manually
await Hive.addFile("./notes.txt");
await Hive.addFile("./photo.jpg");

// Search
const results = await Hive.find("your search query", 10);
console.log(results);
```

## Authentication & RBAC

### Built-in Roles

| Role | Actions |
|------|---------|
| `root` / `dbOwner` | All (`*`) |
| `readWrite` | `find`, `insert`, `update`, `delete`, `embed` |
| `read` | `find`, `embed` |
| `dbAdmin` | `admin` (user management) |

```javascript
await Hive.init({
  dbName: "MyDB",
  permissions: {
    users: {
      admin: { password: "secret", roles: ["dbOwner"] },
      viewer: { password: "view123", roles: ["read"] }
    },
    autoAuth: "admin"
  }
});

Hive.whoAmI(); // "admin"
```

> **No users configured** = all operations allowed (backward compatible).

## API

| Method | Description |
|--------|-------------|
| `Hive.init(options)` | Initialize the database |
| `Hive.embed(input, type)` | Generate a vector embedding |
| `Hive.find(query, topK)` | Vector similarity search |
| `Hive.addFile(filePath)` | Add a file (auto-detects text/image) |
| `Hive.removeFile(filePath)` | Remove file entries |
| `Hive.insertOne(entry)` | Insert a document |
| `Hive.insertMany(entries)` | Bulk insert |
| `Hive.deleteOne(id)` | Delete by ID |
| `Hive.updateOne(query, entry)` | Update a document |
| `Hive.auth(username, password)` | Authenticate a user |
| `Hive.logout()` | Clear current user |
| `Hive.whoAmI()` | Get current username |
| `Hive.createUser({ user, pwd, roles })` | Create a user |
| `Hive.dropUser(username)` | Delete a user |
| `Hive.getUsers()` | List users with roles |

### `Hive.init(options)`

| Option | Default | Description |
|--------|---------|-------------|
| `dbName` | `"Documents"` | Database name |
| `storageDir` | `process.cwd()` | Directory for the database folder |
| `pathToDB` | — | Full path to `.bin` file (overrides `storageDir`) |
| `pathToDocs` | `false` | Directory to auto-import documents from |
| `watch` | `false` | Watch directory for changes |
| `logging` | `false` | Enable processing logs |
| `SliceSize` | `512` | Token limit per chunk |
| `minSliceSize` | `100` | Minimum tokens to index a chunk |
| `overlap` | `5%` of `SliceSize` | Chunk overlap (percentage < 1 or token count >= 1) |
| `rerank` | `false` | Enable cross-encoder reranking |
| `models` | — | Override embedding models (`{ text, image, rerank }`) |
| `permissions` | — | Auth config: `{ users: { user: { password, roles } }, autoAuth }` |

## Python Support

```python
from hive import Hive

with Hive() as hive:
    hive.init({"dbName": "MyDocuments"})
    hive.add_file("./notes.txt")
    results = hive.find("search query")
```

Requires Node.js in PATH and `npm install` in the hive directory.

## MCP Server

Hive includes an MCP server for AI agent integration. All Hive operations are exposed as tools.

```bash
node mcp.js
```

### Tools

`hive_init` · `hive_find` · `hive_insert_one` · `hive_insert_many` · `hive_delete_one` · `hive_update_one` · `hive_add_file` · `hive_remove_file` · `hive_embed` · `hive_auth` · `hive_whoami` · `hive_logout` · `hive_create_user` · `hive_drop_user` · `hive_get_users`

### AI Client Setup

Add to your MCP client config (Cursor, Claude Desktop, opencode, etc.):

```json
{
  "mcpServers": {
    "hive": {
      "command": "node",
      "args": ["/path/to/hive/mcp.js"]
    }
  }
}
```

In **opencode**, use the `hive` skill or configure it in your `opencode.json`:

## Performance

~30ms search in a database with **30,000 entries** on an AMD Ryzen 7 3700X.

## License

MIT
