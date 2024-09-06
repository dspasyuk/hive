const Hive  = require('./hive.js');

async function test() {
    const pathToDocs = './docs';
    await Hive.init('Documents', './db/Documents/db.json', pathToDocs);
    //or 
    // await Hive.init('Documents');
    const vector = await Hive.getVector("Human Factors Workscope", Hive.TransOptions);
    console.time("vector");
    results = await Hive.find('Documents', vector.data, 10);
    console.timeEnd("vector");
    // console.log(results);
    results.forEach((r) => console.log(r.document.meta))
}
test()


// // Load database into memory for search operations
// const db2 = new Hive('myVectorDB');
// db2.loadToMemory();
