import Hive from './hive.js';

const testCases = [
  "hello-world",
  "co-operation",
  "end.Start",
  "one/two",
  "user's data",
  "bread & butter",
  "file_name.txt",
  "100% guaranteed",
  "item #1",
  "foo@bar.com"
];

console.log("Testing Hive.escapeChars behavior:");
testCases.forEach(text => {
  const escaped = Hive.escapeChars(text);
  console.log(`Original: "${text}" -> Escaped: "${escaped}"`);
});
