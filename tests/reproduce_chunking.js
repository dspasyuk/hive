
import Hive from "./hive.js";
import fs from "fs";
import path from "path";

const TEST_FILE = "test_chunking.txt";
const DB_NAME = "TestChunkingDB";

// Create a dummy text file with numbers 1 to 100
const tokens = Array.from({ length: 100 }, (_, i) => (i + 1).toString());
const text = tokens.join(" ");
fs.writeFileSync(TEST_FILE, text);

async function runTest() {
  console.log("--- Testing Chunking Behavior ---");

  // Clean up previous DB
  const dbPath = path.join(process.cwd(), "db", DB_NAME);
  if (fs.existsSync(dbPath)) {
    fs.rmSync(dbPath, { recursive: true, force: true });
  }

  // Initialize Hive with small SliceSize to easily see chunks
  // Default behavior (no overlap expected)
  await Hive.init({
    dbName: DB_NAME,
    type: "text",
    SliceSize: 10, // Small slice size
    minSliceSize: 1,
    overlap: 0.2, // 20% overlap (2 tokens)
    logging: true,
    rerank: false
  });

  // Manually trigger readFile logic (or close enough)
  // Since we can't easily mock doc2txt without changing code, we'll use the actual file
  // and rely on Hive.readFile calling doc2txt.extractTextFromFile.
  // We need to make sure doc2txt can read .txt files.
  
  // We will use a modified approach to inspect chunks without relying on internal logging if possible,
  // but Hive.addItem inserts into the collection. We can inspect the collection after processing.
  
  await Hive.readFile(TEST_FILE, "text");

  const collection = Hive.collections.get(DB_NAME);
  console.log(`Total chunks: ${collection.length}`);

  for (let i = 0; i < collection.length; i++) {
    const content = collection[i].meta.content;
    const firstToken = content.split(" ")[0];
    const lastToken = content.split(" ").pop();
    console.log(`Chunk ${i + 1}: Start '${firstToken}' ... End '${lastToken}'`);
  }

  // Cleanup
  fs.unlinkSync(TEST_FILE);
  // fs.rmSync(dbPath, { recursive: true, force: true });
}

runTest();
