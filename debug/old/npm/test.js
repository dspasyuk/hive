import Hive from './hive.js';

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
const vector = await Hive.getVector("And why should Caesar be a tyrant then?", Hive.TransOptions);
const results = await Hive.find(vector.data, 3);
results.forEach(element => {
    console.log(element.document.meta, element.similarity);
});
