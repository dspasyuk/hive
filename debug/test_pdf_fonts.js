import doc2txt from './doc2txt.js';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function test() {
  const pdfPath = path.join(__dirname, 'doc', '1.2.46.1.Rev.1-Video_Deflector_Modification.pdf');
  console.log(`Testing PDF: ${pdfPath}`);
  try {
      const result = await doc2txt.extractTextFromFile(pdfPath);
      console.log("Text length:", result.text ? result.text.length : 0);
      console.log("Metadata extracted:", !!result.metadata);
      if (result.metadata) {
          console.log("Title:", result.metadata.metadata.title);
      }
  } catch (e) {
      console.error("Test failed:", e);
  }
}

test();
