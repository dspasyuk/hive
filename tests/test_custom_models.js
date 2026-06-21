import Hive from "./hive.js";

async function testCustomModels() {
    console.log("Testing custom models configuration...");

    const customModels = {
        text: "Xenova/all-MiniLM-L6-v2",
        image: "Xenova/clip-vit-base-patch16",
        rerank: "Xenova/ms-marco-MiniLM-L-12-v2"
    };

    await Hive.init({
        dbName: "TestCustomModels",
        models: customModels,
        rerank: true
    });

    let success = true;

    if (Hive.models.text !== customModels.text) {
        console.error(`[FAIL] Text model mismatch. Expected: ${customModels.text}, Got: ${Hive.models.text}`);
        success = false;
    } else {
        console.log(`[PASS] Text model updated correctly: ${Hive.models.text}`);
    }

    if (Hive.models.image !== customModels.image) {
        console.error(`[FAIL] Image model mismatch. Expected: ${customModels.image}, Got: ${Hive.models.image}`);
        success = false;
    } else {
        console.log(`[PASS] Image model updated correctly: ${Hive.models.image}`);
    }

    if (Hive.models.rerank !== customModels.rerank) {
        console.error(`[FAIL] Rerank model mismatch. Expected: ${customModels.rerank}, Got: ${Hive.models.rerank}`);
        success = false;
    } else {
        console.log(`[PASS] Rerank model updated correctly: ${Hive.models.rerank}`);
    }

    if (success) {
        console.log("All custom model tests passed!");
    } else {
        console.error("Some tests failed.");
        process.exit(1);
    }
}

testCustomModels();
