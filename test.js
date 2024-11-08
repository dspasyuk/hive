import Hive from '@deonis/hive';

const options = {
    dbName: "Documents",
    pathToDB: "./db/Documents/db.json",
    pathToDocs: "./docs",
    type: "text",
    documents: {
      text: [".txt", ".doc", ".docx", ".pdf"],
      image: [".png", ".jpg", ".jpeg"]
    }
  };
await Hive.init(options); 
const vector = await Hive.getVector("Tut, I have lost myself; I am not here; This is not Romeo, he's some other where.", Hive.TransOptions);
const imgvector = await Hive.getVector("./docs/Smithsonian/smithm352e60b53736d45668efde4e22339f424c32c.jpg", Hive.TransOptions);
var results = await Hive.find(imgvector.data, 3);
results.forEach(element => {
     console.log(element.document.meta, element.similarity);
});

results = await Hive.find(vector.data, 3);
results.forEach(element => {
    console.log(element.document.meta, element.similarity);
});
