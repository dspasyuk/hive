import Hive from "./hive.js"; // Assuming your file is named Hive.js
import fs from "fs";
import path from "path";

// Mocking the external dependencies for isolated timing
// The pipeline mock should return a function that returns a structure similar to the real one.
const mockVector = new Float32Array(384).fill(0.123); // A realistic size for all-MiniLM-L6-v2

const mockPipeline = async (task, model) => {
    // Simulate the async loading time of the model
    await new Promise(resolve => setTimeout(resolve, 50)); 
    return async (input, options) => {
        // Simulate the embedding generation time
        await new Promise(resolve => setTimeout(resolve, 5)); 
        return { data: mockVector }; // Mocked output structure
    };
};

// Mock the doc2txt library for file parsing
const mockDoc2Txt = {
    extractTextFromFile: async (filePath) => {
        // Simulate file parsing time
        await new Promise(resolve => setTimeout(resolve, 10)); 
        return { text: "This is a sample document content for testing purposes and speed checks." };
    }
};

// Replace original imports with mocks/dummies for testing
Hive.pipeline = mockPipeline; // Overwrite the pipeline import reference
// You might need to adjust the path to your mock/real doc2txt if necessary
// Assuming doc2txt is imported directly in Hive.js, we'll manually mock the function call inside Hive.addFile/Hive.readFile if needed.
// For simplicity here, we'll just mock the entire readFile/addFile chain or ensure they use the dummy.

// --- Helper for Timing and Logging ---
const timers = {};

async function timeFunction(name, func, ...args) {
    const start = process.hrtime.bigint();
    let result;
    try {
        result = await func(...args);
    } catch (error) {
        console.error(`Error in function ${name}:`, error.message);
        throw error;
    }
    const end = process.hrtime.bigint();
    const durationMs = Number(end - start) / 1000000;
    timers[name] = durationMs;
    console.log(`⏱️ ${name}: ${durationMs.toFixed(3)} ms`);
    return result;
}

// --- Setup Test Data and Environment ---

// Use a temporary test path
const TEST_DB_DIR = path.join(process.cwd(), "test_db_temp");
const TEST_DB_PATH = path.join(TEST_DB_DIR, "TestDB.json");

function cleanup() {
    if (fs.existsSync(TEST_DB_DIR)) {
        fs.rmSync(TEST_DB_DIR, { recursive: true, force: true });
    }
}

// Ensure the Hive instance is clean before each test block
function resetHive() {
    Hive.collections.clear();
    Hive.users.clear();
    Hive.currentUser = null;
    Hive.pipeline = null;
    Hive.dbName = "TestDB";
    Hive.pathToDB = TEST_DB_PATH;
    Hive.saveTimeout = null;
    // Set a very short interval for tests to avoid long waits
    Hive.saveInterval = 100; 
}


// --- Test Functions ---

async function testInitAndEmbedding() {
    console.log("\n## 1. Initialization and Embedding Speed Tests");
    resetHive();
    cleanup(); // Clean before init

    // 1.1 Test init (includes model loading/simulated load)
    await timeFunction("Hive.init_TextMode", Hive.init, { 
        dbName: "TestDB", 
        pathToDB: TEST_DB_PATH,
        pathToDocs: false, // Don't pull documents on init for this test
        type: "text" 
    });
    
    // 1.2 Test textEmbeddingInit (should be fast if pipeline is already set)
    await timeFunction("Hive.textEmbeddingInit_Repeated", Hive.textEmbeddingInit); 

    // 1.3 Test addItem (includes vector generation, mocked)
    await timeFunction("Hive.addItem_SingleText", Hive.addItem, "A short test sentence to be converted to a vector.");
    await new Promise(resolve => setTimeout(resolve, Hive.saveInterval + 50)); // Wait for auto-save

    // 1.4 Test insertOne (core logic only, vector generation is separate)
    resetHive();
    Hive.createCollection();
    const mockEntry = { vector: mockVector, meta: { source: "manual" } };
    await timeFunction("Hive.insertOne_CoreLogic", Hive.insertOne, mockEntry);
    await new Promise(resolve => setTimeout(resolve, Hive.saveInterval + 50)); // Wait for auto-save
}

async function testDiskOperations() {
    console.log("\n## 2. Disk Operations Speed Tests");
    resetHive();
    cleanup();
    Hive.createCollection();
    
    // Create a large collection of dummy data (1000 entries)
    const BULK_SIZE = 1000;
    const bulkData = [];
    for (let i = 0; i < BULK_SIZE; i++) {
        bulkData.push({ vector: mockVector, meta: { id: i, source: "bulk" } });
    }
    Hive.insertMany(bulkData); // This also schedules a saveToDisk

    // Wait for the scheduled save
    await new Promise(resolve => setTimeout(resolve, Hive.saveInterval + 50)); 
    console.log(`Database primed with ${BULK_SIZE} entries for load test.`);

    // 2.1 Test saveToDisk (manual call, should be fast if auto-save just finished, but we'll test the actual write)
    await timeFunction("Hive.saveToDisk_1MEntries", Hive.saveToDisk); 
    await new Promise(resolve => setTimeout(resolve, Hive.saveInterval + 50)); // Wait for final save

    // 2.2 Test loadToMemory
    Hive.collections.clear(); // Clear memory to simulate fresh load
    await timeFunction("Hive.loadToMemory_1MEntries", Hive.loadToMemory);
}

async function testQueryAndLogic() {
    console.log("\n## 3. Query and Core Logic Speed Tests");
    resetHive();
    Hive.createCollection();

    // Setup for search: Add 10,000 items for a good stress test
    const SEARCH_SIZE = 1000000;
    const searchData = [];
    for (let i = 0; i < SEARCH_SIZE; i++) {
        // Create slightly different vectors for realistic cosine similarity
        const vector = new Float32Array(mockVector);
        vector[i % vector.length] = Math.random(); 
        searchData.push({ vector, meta: { id: i, type: "search" } });
    }
    Hive.insertMany(searchData);
    console.log(`Collection primed with ${SEARCH_SIZE} entries for search test.`);

    // 3.1 Test normalize
    await timeFunction("Hive.normalize", Hive.normalize, mockVector);

    // 3.2 Test cosineSimilarity
    await timeFunction("Hive.cosineSimilarity", Hive.cosineSimilarity, mockVector, mockVector, Hive.normalize(mockVector), Hive.normalize(mockVector));

    // 3.3 Test find (vector search)
    const queryVector = new Float32Array(mockVector);
    queryVector[0] = 1.0; // Make the query slightly unique
    await timeFunction("Hive.find_1MItems_Top10", Hive.find, queryVector, 10);
    
    // 3.4 Test tokenCount
    const testText = "The quick brown fox jumps over the lazy dog. TEY FY AFRL 12345.";
    await timeFunction("Hive.tokenCount", Hive.tokenCount, testText);

    // 3.5 Test escapeChars
    const dirtyText = `A "test" string & with <special> chars \\ and 'quotes' / and TEY FY 1234. R R R A A A.`;
    await timeFunction("Hive.escapeChars", Hive.escapeChars, dirtyText);
}

async function testAuthAndPermissions() {
    console.log("\n## 4. Authentication & Access Control Tests");

    // 4.1 Backward compatible: no users = no auth required
    console.log("\n4.1 No users configured - operations proceed without auth");
    resetHive();
    Hive.createCollection();
    Hive.insertOne({ vector: [1, 0], meta: { test: true } });
    const findResult = await Hive.find([1, 0], 5);
    console.assert(findResult.length > 0, "find should work without auth");
    console.log("   ✓ insert/find work without users configured");

    // 4.2 Bootstrap: create first user (no auth needed when no users exist)
    console.log("\n4.2 Bootstrap - create first user without auth");
    resetHive();
    Hive.createCollection();
    Hive.createUser({ user: "admin", pwd: "admin123", roles: ["dbOwner"] });
    console.assert(Hive.users.size === 1, "should have 1 user");
    console.log("   ✓ first user created without auth");

    // 4.3 Operations blocked without auth when users exist
    console.log("\n4.3 Operations blocked without auth when users exist");
    try {
        Hive.insertOne({ vector: [1, 0], meta: {} });
        console.assert(false, "should have thrown");
    } catch (e) {
        console.assert(e.message.includes("Not authenticated"), "wrong error: " + e.message);
        console.log("   ✓ insert blocked without auth");
    }
    try {
        await Hive.find([1, 0], 5);
        console.assert(false, "should have thrown");
    } catch (e) {
        console.assert(e.message.includes("Not authenticated"), "wrong error: " + e.message);
        console.log("   ✓ find blocked without auth");
    }

    // 4.4 Authentication with wrong password
    console.log("\n4.4 Authentication with wrong password");
    try {
        Hive.auth("admin", "wrongpass");
        console.assert(false, "should have thrown");
    } catch (e) {
        console.assert(e.message.includes("Invalid credentials"), "wrong error: " + e.message);
        console.log("   ✓ wrong password rejected");
    }

    // 4.5 Authentication with correct password
    console.log("\n4.5 Authentication with correct password");
    Hive.auth("admin", "admin123");
    console.assert(Hive.whoAmI() === "admin", "whoAmI should return admin");
    console.log("   ✓ authenticated as admin");

    // 4.6 Admin can perform all operations
    console.log("\n4.6 Admin can perform all operations");
    Hive.insertOne({ vector: [1, 0], meta: { title: "admin test" } });
    const adminFind = await Hive.find([1, 0], 5);
    console.assert(adminFind.length > 0, "admin should be able to find");
    console.log("   ✓ admin insert/find works");

    // 4.7 Create users with different roles
    console.log("\n4.7 Create users with different roles");
    Hive.createUser({ user: "reader", pwd: "read123", roles: ["read"] });
    Hive.createUser({ user: "writer", pwd: "write123", roles: ["readWrite"] });
    Hive.createUser({ user: "dba", pwd: "dba123", roles: ["dbAdmin"] });
    console.assert(Hive.users.size === 4, "should have 4 users");
    console.log("   ✓ 3 additional users created");

    // 4.8 getUsers requires admin
    console.log("\n4.8 getUsers requires admin privilege");
    const userList = Hive.getUsers();
    console.assert(userList.length === 4, "should list 4 users");
    console.log("   ✓ admin can list users");

    // 4.9 Test read-only role
    console.log("\n4.9 Read-only role restrictions");
    Hive.auth("reader", "read123");
    console.assert(Hive.whoAmI() === "reader", "whoAmI should return reader");
    const readerFind = await Hive.find([1, 0], 5);
    console.assert(readerFind.length > 0, "reader should be able to find");
    console.log("   ✓ reader can find");
    try {
        Hive.insertOne({ vector: [2, 0], meta: {} });
        console.assert(false, "should have thrown");
    } catch (e) {
        console.assert(e.message.includes("not authorized"), "wrong error: " + e.message);
        console.log("   ✓ reader cannot insert");
    }
    try {
        Hive.deleteOne("some-id");
        console.assert(false, "should have thrown");
    } catch (e) {
        console.assert(e.message.includes("not authorized"), "wrong error: " + e.message);
        console.log("   ✓ reader cannot delete");
    }

    // 4.10 Test readWrite role
    console.log("\n4.10 ReadWrite role permissions");
    Hive.auth("writer", "write123");
    Hive.insertOne({ vector: [3, 0], meta: { title: "writer test" } });
    console.log("   ✓ writer can insert");
    const writerFind = await Hive.find([3, 0], 5);
    console.assert(writerFind.length > 0, "writer should be able to find");
    console.log("   ✓ writer can find");
    try {
        Hive.createUser({ user: "extra", pwd: "pass", roles: ["read"] });
        console.assert(false, "should have thrown");
    } catch (e) {
        console.assert(e.message.includes("not authorized"), "wrong error: " + e.message);
        console.log("   ✓ writer cannot create users (admin only)");
    }

    // 4.11 Test dbAdmin role
    console.log("\n4.11 dbAdmin role can manage users");
    Hive.auth("dba", "dba123");
    Hive.getUsers();
    console.log("   ✓ dbAdmin can list users");
    Hive.createUser({ user: "extra", pwd: "pass", roles: ["read"] });
    console.assert(Hive.users.has("extra"), "extra user should exist");
    console.log("   ✓ dbAdmin can create users");
    try {
        Hive.insertOne({ vector: [4, 0], meta: {} });
        console.assert(false, "should have thrown");
    } catch (e) {
        console.assert(e.message.includes("not authorized"), "wrong error: " + e.message);
        console.log("   ✓ dbAdmin cannot insert data");
    }

    // 4.12 Logout and whoAmI
    console.log("\n4.12 Logout functionality");
    Hive.logout();
    console.assert(Hive.whoAmI() === null, "whoAmI should return null after logout");
    console.log("   ✓ logout clears current user");

    // 4.13 dropUser
    console.log("\n4.13 dropUser (re-authenticate as admin first)");
    Hive.auth("admin", "admin123");
    Hive.dropUser("extra");
    console.assert(!Hive.users.has("extra"), "extra user should be deleted");
    console.log("   ✓ user dropped successfully");
    try {
        Hive.dropUser("admin");
        console.assert(false, "should have thrown");
    } catch (e) {
        console.assert(e.message.includes("Cannot delete yourself"), "wrong error: " + e.message);
        console.log("   ✓ cannot delete yourself");
    }

    console.log("\n   ✓ All auth & permissions tests passed");
}

// --- Main Test Execution ---

async function runTests() {
    console.log("--- Hive Database Performance Test Script ---");
    try {
        await testInitAndEmbedding();
        await testDiskOperations();
        await testQueryAndLogic();
        await testAuthAndPermissions();
    } catch (e) {
        console.error("\n💥 A critical error occurred during testing!");
        console.error(e);
    } finally {
        // Final cleanup
        cleanup();
        resetHive();
        console.log("\n--- Tests Complete ---");
        console.log("Summary of Function Speeds (ms):", timers);
    }
}

runTests();
