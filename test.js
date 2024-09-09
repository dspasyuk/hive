import Hive from './hive.js';

// async function test() {
await Hive.init('Documents', './db/Documents/db.json',"./docs");
// await db.initTransformers();
// db.loadToMemory();  
const vector = await Hive.getVector("Human Factors Workscope", Hive.TransOptions);
for (let i = 0; i < 10; i++) {
    console.time("vector");
    var results = await Hive.find(vector.data, 10);
    console.timeEnd("vector");    
}
   
// }
// test()


// // Load database into memory for search operations
// const db2 = new Hive('myVectorDB');
// db2.loadToMemory();
