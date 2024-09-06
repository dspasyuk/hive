const Hive  = require('./hive.js');

async function test() {
    await Hive.init('documents', './db/Documents/db.json');
    // await db.initTransformers();
    // db.loadToMemory();  
    const vector = await Hive.getVector("Human Factors Workscope", Hive.TransOptions);
    console.time("vector");
    results = Hive.find('Documents', vector.data, 10);
    console.timeEnd("vector");
    // console.log(results);
    results.forEach((r) => console.log(r.document.meta.href, r.similarity))
    console.time("vector");
    results = Hive.find('Documents', vector.data, 10);
    console.timeEnd("vector");
}
test()


// // Load database into memory for search operations
// const db2 = new Hive('myVectorDB');
// db2.loadToMemory();
