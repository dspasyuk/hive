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
        SliceSize: { type: "number", description: "Token limit for text slicing (default: 512)" },
        overlap: { type: "number", description: "Chunk overlap in tokens or percentage" },
        rerank: { type: "boolean", description: "Enable cross-encoder reranking" }
      }
    }
  },
  {
    name: "hive_find",
    description: "Vector similarity search",
    inputSchema: {
      type: "object",
      properties: {
        query: { oneOf: [{ type: "string" }, { type: "array", items: { type: "number" } }], description: "Text query or vector array" },
        topK: { type: "number", description: "Number of results (default: 10)" }
      },
      required: ["query"]
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
  }
];

function ok(data) {
  return { content: [{ type: "text", text: JSON.stringify(data, null, 2) }] };
}

function err(msg) {
  return { content: [{ type: "text", text: msg }], isError: true };
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
        const results = await Hive.find(args.query, args.topK || 10);
        return ok(results.map(r => ({
          id: r.document.id, similarity: r.similarity, meta: r.document.meta
        })));
      }
      case "hive_add_file": {
        await Hive.addFile(args.filePath);
        return ok({ message: `File "${args.filePath}" added` });
      }
      case "hive_embed": {
        const vector = await Hive.embed(args.input, args.type || "text");
        return ok({ vector, dimensions: vector.length });
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
