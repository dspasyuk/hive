
import Hive from './hive.js';
import fs from 'fs';
import path from 'path';

async function test() {
    console.log("Starting Multimodal Hive Test...");

    // Setup dummy files
    const testTextFile = path.resolve('test_doc.txt');
    fs.writeFileSync(testTextFile, "The quick brown fox jumps over the lazy dog.");

    // Initialize Hive
    console.log("Initializing Hive...");
    await Hive.init({
        dbName: "TestDB",
        pathToDB: path.resolve('db/TestDB/TestDB.json'),
        pathToDocs: false, // We will add files manually
        logging: true, // Enable logging to test
    });

    // Add Text File
    console.log("Adding Text File...");
    await Hive.addFile(testTextFile);

    // Verify Text Search
    console.log("Searching for 'fox'...");
    const textEmbedding = await Hive.embed("fox", "text");
    const textResults = await Hive.find(textEmbedding);
    console.log("Text Results:", textResults.map(r => r.document.meta.content));

    if (textResults.length > 0 && textResults[0].document.meta.type === 'text') {
        console.log("✅ Text search successful");
    } else {
        console.error("❌ Text search failed");
    }

    // Add Image File (if available)
    const imgDir = path.resolve('img/doc');
    if (fs.existsSync(imgDir)) {
        const files = fs.readdirSync(imgDir);
        const imageFile = files.find(f => ['.jpg', '.png', '.jpeg'].includes(path.extname(f).toLowerCase()));
        
        if (imageFile) {
            const imagePath = path.join(imgDir, imageFile);
            console.log(`Adding Image File: ${imageFile}...`);
            await Hive.addFile(imagePath);

            // Verify Image Search (using the image itself as query for simplicity, or text if we had a text-to-image search which we don't fully have yet without cross-modal)
            // Wait, CLIP allows text-to-image search if we use the text encoder for query and image encoder for items.
            // But here Hive.find expects vectors.
            // If I embed text "a photo of a rock" using text pipeline, and compare to image vector?
            // The current implementation uses separate pipelines. 
            // Xenova/clip-vit-base-patch32 has both text and vision models.
            // My implementation uses `feature-extraction` for text (all-MiniLM) and `image-feature-extraction` for image (clip).
            // These are DIFFERENT SPACES. So they won't match.
            // So I can only search Image-to-Image or Text-to-Text with current setup unless I switch text model to CLIP text encoder.
            // The user asked: "Can we generate embedding for any supported file format and switch between them without the need to specify in options image or text?"
            // And "handles either text or image could we make it indiferent".
            
            // For now, I will test Image-to-Image search (using the same image as query).
            console.log("Searching for Image (using same image)...");
            const imageEmbedding = await Hive.embed(imagePath, "image");
            const imageResults = await Hive.find(imageEmbedding);
            console.log("Image Results:", imageResults.map(r => r.document.meta.filePath));

            if (imageResults.length > 0 && imageResults[0].document.meta.type === 'image') {
                console.log("✅ Image search successful");
            } else {
                console.error("❌ Image search failed");
            }
        } else {
            console.log("⚠️ No image files found in img directory to test.");
        }
    }

    // Cleanup
    if (fs.existsSync(testTextFile)) fs.unlinkSync(testTextFile);
    // fs.rmSync(path.resolve('db/TestDB'), { recursive: true, force: true });
}

test().catch(console.error);
