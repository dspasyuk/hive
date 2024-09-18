import Hive from 'deonis/hive';
await Hive.init('Documents', "./DB/Documents/db.js", "./doc", );

const vector = await Hive.getVector("Hello World", Hive.TransOptions);
const imgvector = await Hive.getVector("path/to/test/image.jpg", Hive.TransOptions);
var results = await Hive.find(imgvector.data, 3);
results.forEach(element => {
     console.log(element.document.meta, element.similarity);
});

results = await Hive.find(vector.data, 3);
results.forEach(element => {
    console.log(element.document.meta);
});
