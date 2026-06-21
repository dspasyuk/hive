//import Hive from '@deonis/hive';
import Hive from './hive.js'
await Hive.init({dbName:'Documents', pathToDocs:"./img"});

const vector = await Hive.getVector("/home/denis/akatoreite65262b.JPG", Hive.TransOptions);
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
