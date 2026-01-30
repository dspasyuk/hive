#!/usr/bin/env node
import Hive from './hive.js';
import readline from 'readline';

// Redirect console.log to stderr so it doesn't interfere with stdout JSON communication
console.log = console.error;

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false
});

/**
 * Send JSON response to stdout
 * @param {any} data 
 * @param {string|null} error 
 */
function sendResponse(data, error = null) {
    const response = JSON.stringify({ data, error });
    process.stdout.write(response + '\n');
}

rl.on('line', async (line) => {
    if (!line.trim()) return;
    
    try {
        const command = JSON.parse(line);
        const { action, args } = command;

        switch (action) {
            case 'init':
                await Hive.init(args || {});
                sendResponse("Hive initialized");
                break;
            
            case 'addFile':
                await Hive.addFile(args.filePath);
                sendResponse("File added");
                break;
            
            case 'embed':
                try {
                    const vector = await Hive.embed(args.input, args.type);
                     // Convert Float32Array to regular array for JSON serialization
                    sendResponse(Array.from(vector));
                } catch (e) {
                     sendResponse(null, e.message);
                }
                break;

            case 'find':
                 try {
                    const results = await Hive.find(args.queryVector, args.topK);
                    // Ensure results are JSON serializable (convert Float32Arrays if any remain)
                    const serializableResults = results.map(r => ({
                        ...r,
                        document: {
                            ...r.document,
                            vector: Array.from(r.document.vector || []),
                            magnitude: r.document.magnitude
                        }
                    }));
                    sendResponse(serializableResults);
                 } catch (e) {
                    sendResponse(null, e.message);
                 }
                break;
            
            case 'insertOne':
                Hive.insertOne(args.entry);
                sendResponse("Inserted");
                break;

            case 'deleteOne':
                Hive.deleteOne(args.id);
                sendResponse("Deleted");
                break;

            case 'updateOne':
                Hive.updateOne(args.query, args.entry);
                sendResponse("Updated");
                break;

            case 'removeFile':
                Hive.removeFile(args.filePath);
                sendResponse("File removed");
                break;

            default:
                sendResponse(null, `Unknown action: ${action}`);
        }

    } catch (error) {
        sendResponse(null, error.message);
    }
});
