#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import Hive from "./hive.js";

const TOOLS = [
  {
    name: "hive_init",
    description: "Initialize the Hive vector database",
    inputSchema: {
      type: "object",
      properties: {
        dbName: { type: "string", description: "Database name (default: Documents)" },
        pathToDocs: { type: "string", description: "Directory to auto-import documents from" },
        pathToDB: { type: "string", description: "Full path to database file" },
        storageDir: { type: "string", description: "Directory to store database" },
        logging: { type: "boolean", description: "Enable logging" },
        watch: { type: "boolean", description: "Watch document directory for changes" },
        SliceSize: { type: "number", description: "Token limit for text slicing (default: 512)" },
        overlap: { type: "number", description: "Chunk overlap in tokens or percentage" },
        rerank: { type: "boolean", description: "Enable cross-encoder reranking" },
        permissions: {
          type: "object",
          description: "Authentication config",
          properties: {
            users: {
              type: "object",
              description: "Map of username to { password, roles }",
              additionalProperties: {
                type: "object",
                properties: {
                  password: { type: "string" },
                  roles: { type: "array", items: { type: "string" } }
                }
              }
            },
            autoAuth: { type: "string", description: "Username to auto-authenticate" }
          }
        }
      }
    }
  },
  {
    name: "hive_find",
    description: "Vector similarity search",
    inputSchema: {
      type: "object",
      properties: {
        query: {
          oneOf: [
            { type: "string", description: "Text query (auto-embedded)" },
            { type: "array", items: { type: "number" }, description: "Raw vector array" }
          ],
          description: "Text query or vector array"
        },
        topK: { type: "number", description: "Number of results (default: 10)" }
      },
      required: ["query"]
    }
  },
  {
    name: "hive_insert_one",
    description: "Insert a document into the database",
    inputSchema: {
      type: "object",
      properties: {
        vector: { type: "array", items: { type: "number" }, description: "Vector embedding" },
        meta: { type: "object", description: "Metadata (title, content, filePath, etc.)" }
      },
      required: ["vector", "meta"]
    }
  },
  {
    name: "hive_insert_many",
    description: "Bulk insert documents",
    inputSchema: {
      type: "object",
      properties: {
        entries: {
          type: "array",
          items: {
            type: "object",
            properties: {
              vector: { type: "array", items: { type: "number" } },
              meta: { type: "object" }
            },
            required: ["vector", "meta"]
          }
        }
      },
      required: ["entries"]
    }
  },
  {
    name: "hive_delete_one",
    description: "Delete a document by ID",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string", description: "Document ID" }
      },
      required: ["id"]
    }
  },
  {
    name: "hive_update_one",
    description: "Update a document matching a query",
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "object",
          description: "Query to match (e.g. { filePath: '...' })",
          properties: {
            filePath: { type: "string" }
          }
        },
        entry: {
          type: "object",
          properties: {
            vector: { type: "array", items: { type: "number" } },
            meta: { type: "object" }
          },
          required: ["vector", "meta"]
        }
      },
      required: ["query", "entry"]
    }
  },
  {
    name: "hive_add_file",
    description: "Add a file to the database (auto-detects text/image)",
    inputSchema: {
      type: "object",
      properties: {
        filePath: { type: "string", description: "Path to the file" }
      },
      required: ["filePath"]
    }
  },
  {
    name: "hive_remove_file",
    description: "Remove all entries for a file path",
    inputSchema: {
      type: "object",
      properties: {
        filePath: { type: "string", description: "File path to remove" }
      },
      required: ["filePath"]
    }
  },
  {
    name: "hive_embed",
    description: "Generate a vector embedding for text or image",
    inputSchema: {
      type: "object",
      properties: {
        input: { type: "string", description: "Text content or image file path" },
        type: { type: "string", enum: ["text", "image"], description: "Input type (default: text)" }
      },
      required: ["input"]
    }
  },
  {
    name: "hive_auth",
    description: "Authenticate a user",
    inputSchema: {
      type: "object",
      properties: {
        username: { type: "string" },
        password: { type: "string" }
      },
      required: ["username", "password"]
    }
  },
  {
    name: "hive_whoami",
    description: "Get the currently authenticated username",
    inputSchema: { type: "object", properties: {} }
  },
  {
    name: "hive_logout",
    description: "Clear the current authenticated user",
    inputSchema: { type: "object", properties: {} }
  },
  {
    name: "hive_create_user",
    description: "Create a new database user (requires admin or no users exist)",
    inputSchema: {
      type: "object",
      properties: {
        user: { type: "string" },
        pwd: { type: "string" },
        roles: { type: "array", items: { type: "string" }, description: "Roles (default: ['read'])" }
      },
      required: ["user", "pwd"]
    }
  },
  {
    name: "hive_drop_user",
    description: "Delete a user (requires admin)",
    inputSchema: {
      type: "object",
      properties: {
        username: { type: "string" }
      },
      required: ["username"]
    }
  },
  {
    name: "hive_get_users",
    description: "List all users with roles (requires admin)",
    inputSchema: { type: "object", properties: {} }
  },
  {
    name: "hive_begin_session",
    description: "Start a transaction session — defers disk saves until commit or rollback",
    inputSchema: { type: "object", properties: {} }
  },
  {
    name: "hive_commit_session",
    description: "Commit the current session — flush all pending changes to disk",
    inputSchema: { type: "object", properties: {} }
  },
  {
    name: "hive_rollback_session",
    description: "Rollback the current session — discard in-memory changes and reload from disk",
    inputSchema: { type: "object", properties: {} }
  }
];

function ok(data) {
  return {
    content: [{ type: "text", text: JSON.stringify(data, null, 2) }]
  };
}

function err(msg) {
  return {
    content: [{ type: "text", text: msg }],
    isError: true
  };
}

const server = new Server(
  { name: "hive-mcp", version: "1.4.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  try {
    switch (name) {
      case "hive_init": {
        await Hive.init(args || {});
        return ok({ message: `Database "${Hive.dbName}" initialized` });
      }
      case "hive_find": {
        const { query, topK } = args;
        const results = await Hive.find(query, topK || 10);
        return ok(results.map(r => ({
          id: r.document.id,
          similarity: r.similarity,
          meta: r.document.meta
        })));
      }
      case "hive_insert_one": {
        Hive.insertOne(args);
        return ok({ message: "Document inserted" });
      }
      case "hive_insert_many": {
        Hive.insertMany(args.entries);
        return ok({ message: `${args.entries.length} documents inserted` });
      }
      case "hive_delete_one": {
        Hive.deleteOne(args.id);
        return ok({ message: `Document "${args.id}" deleted` });
      }
      case "hive_update_one": {
        Hive.updateOne(args.query, args.entry);
        return ok({ message: "Document updated" });
      }
      case "hive_add_file": {
        await Hive.addFile(args.filePath);
        return ok({ message: `File "${args.filePath}" added` });
      }
      case "hive_remove_file": {
        Hive.removeFile(args.filePath);
        return ok({ message: `File "${args.filePath}" removed` });
      }
      case "hive_embed": {
        const vector = await Hive.embed(args.input, args.type || "text");
        return ok({ vector, dimensions: vector.length });
      }
      case "hive_auth": {
        Hive.auth(args.username, args.password);
        return ok({ message: `Authenticated as "${args.username}"` });
      }
      case "hive_whoami": {
        const user = Hive.whoAmI();
        return ok({ user });
      }
      case "hive_logout": {
        Hive.logout();
        return ok({ message: "Logged out" });
      }
      case "hive_create_user": {
        Hive.createUser({ user: args.user, pwd: args.pwd, roles: args.roles || ["read"] });
        return ok({ message: `User "${args.user}" created` });
      }
      case "hive_drop_user": {
        Hive.dropUser(args.username);
        return ok({ message: `User "${args.username}" dropped` });
      }
      case "hive_get_users": {
        const users = Hive.getUsers();
        return ok({ users });
      }
      case "hive_begin_session": {
        Hive.beginSession();
        return ok({ message: "Session started — disk saves deferred" });
      }
      case "hive_commit_session": {
        await Hive.commitSession();
        return ok({ message: "Session committed — all changes saved to disk" });
      }
      case "hive_rollback_session": {
        await Hive.rollbackSession();
        return ok({ message: "Session rolled back — in-memory state reverted from disk" });
      }
      default:
        return err(`Unknown tool: ${name}`);
    }
  } catch (e) {
    return err(e.message);
  }
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch(console.error);
