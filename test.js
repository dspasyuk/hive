import Hive from './hive.js';

await Hive.init('Documents', './db/Documents/db.json',"./docs");
const vector = await Hive.getVector("Human Factors Workscope", Hive.transformersOptions);
var results = await Hive.find(vector.data, 10);
results.forEach(element => {
     console.log(element.document.meta, element.similarity);
});
