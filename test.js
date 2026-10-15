//import Hive from '@deonis/hive';
import Hive from './hive.js'
await Hive.init({dbName:'Documents', pathToDocs:"./doc"});

const vector = await Hive.getVector("Tut, I have lost myself; I am not here; This is not Romeo, he's some other where.", Hive.TransOptions);
var results;
console.time("find");
results = await Hive.find(vector.data, 3);
results.forEach(element => {
    console.log(element.document.meta, element.similarity);
});
console.timeEnd("find");
console.time("find");
results = await Hive.find(vector.data, 3);
results.forEach(element => {
    console.log(element.document.meta, element.similarity);
});
console.timeEnd("find");