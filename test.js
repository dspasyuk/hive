import Hive from './hive.js';

await Hive.init('Documents', './db/Documents/db.json',"./docs");
const vector = await Hive.getVector("Human Factors Workscope", Hive.TransOptions);
var results = await Hive.find(vector.data, 10);
