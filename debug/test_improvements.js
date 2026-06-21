import Hive from "./hive.js";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const testDBPath = path.join(__dirname, "test_db_improvements", "TestDB.json");

async function runTests() {
  console.log("Starting Improvement Tests...");

  // Clean up previous test
  if (fs.existsSync(path.dirname(testDBPath))) {
    fs.rmSync(path.dirname(testDBPath), { recursive: true, force: true });
  }

  // Test 1: Watch option respect
  console.log("Test 1: Checking watch option...");
  await Hive.init({
    dbName: "TestDB",
    pathToDB: testDBPath,
    watch: true, // Should be respected now
    pathToDocs: false
  });

  if (Hive.watch !== true) {
    console.error("FAILED: Hive.watch should be true");
    process.exit(1);
  } else {
    console.log("PASSED: Hive.watch is true");
  }

  // Test 2: Basic Insert and Save (Atomic Write Check - indirect)
  console.log("Test 2: Basic Insert and Save...");
  await Hive.addItem("Hello World", "test.txt", "text");
  
  // Wait for save (debounce is 5000ms, we can force save or wait)
  // We'll force save for test speed if possible, but saveToDisk is debounced.
  // Let's wait slightly more than 5s or just check memory state first.
  
  if (Hive.collections.get("TestDB").length !== 1) {
     console.error("FAILED: Collection should have 1 item");
     process.exit(1);
  }
  console.log("PASSED: Item added to memory");

  console.log("Waiting for auto-save (5s)...");
  await new Promise(resolve => setTimeout(resolve, 6000));

  if (fs.existsSync(testDBPath)) {
      const content = fs.readFileSync(testDBPath, 'utf8');
      if (content.includes("Hello World")) {
          console.log("PASSED: Data saved to disk");
      } else {
          console.error("FAILED: Data not found on disk");
          process.exit(1);
      }
  } else {
      console.error("FAILED: DB file not created");
      process.exit(1);
  }

  console.log("All Improvement Tests Passed!");
}

runTests().catch(console.error);
