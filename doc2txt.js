#!/usr/bin/env node
import path from 'path';
import JSZip from 'jszip';
import fs from 'fs/promises';
import { parseStringPromise } from 'xml2js';
import WordExtractor from 'word-extractor';
import { readPdfText } from 'pdf-text-reader';
import * as pdfjsLib from 'pdfjs-dist';
import { fileURLToPath } from 'url';
import { createCanvas, ImageData } from 'canvas';
// ==================================================================
// 🔧 POLYFILL: Fix for "Promise.withResolvers is not a function"
// ==================================================================
if (typeof Promise.withResolvers === "undefined") {
  Promise.withResolvers = function () {
    let resolve, reject;
    const promise = new Promise((res, rej) => {
      resolve = res;
      reject = rej;
    });
    return { promise, resolve, reject };
  };
}



const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const standardFontDataUrl = path.join(__dirname, 'node_modules/pdfjs-dist/standard_fonts/');

// ------------------------------------------------------------------
// HELPER: Timeout Wrapper
// Prevents any single operation (text or image) from hanging forever
// ------------------------------------------------------------------
const runWithTimeout = (promise, ms, label = "Operation") => {
  let timeoutId;
  const timeoutPromise = new Promise((_, reject) => {
    timeoutId = setTimeout(() => {
      reject(new Error(`${label} timed out after ${ms}ms`));
    }, ms);
  });

  return Promise.race([promise, timeoutPromise]).finally(() => {
    clearTimeout(timeoutId);
  });
};

async function rgbaToPngBase64(image) {
  const { width, height, data } = image;
  // Safety check for empty or zero-size images
  if (width === 0 || height === 0 || !data) return null;

  let rgba;
  if (data.length === width * height * 3) {
    rgba = new Uint8ClampedArray(width * height * 4);
    for (let i = 0, j = 0; i < data.length; i += 3, j += 4) {
      rgba[j] = data[i]; rgba[j + 1] = data[i+1]; rgba[j + 2] = data[i+2]; rgba[j + 3] = 255;
    }
  } else if (data.length === width * height) {
    rgba = new Uint8ClampedArray(width * height * 4);
    for (let i = 0, j = 0; i < data.length; i++, j += 4) {
      const val = data[i];
      rgba[j] = rgba[j+1] = rgba[j+2] = val;
      rgba[j+3] = 255;
    }
  } else {
    rgba = new Uint8ClampedArray(data);
  }

  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  const imgData = new ImageData(rgba, width, height);
  ctx.putImageData(imgData, 0, 0);

  return canvas.toDataURL('image/png');
}

pdfjsLib.GlobalWorkerOptions.workerSrc = 'pdfjs-dist/build/pdf.worker.mjs';

const doc2txt = {};

doc2txt.readTextFromTxt = async function (filePath) {
  try {
    return await fs.readFile(filePath, 'utf8');
  } catch (err) {
    console.error("Error reading .txt file:", err.message);
    return "";
  }
};

doc2txt.extractPdfMetadata = async function(pdfUrl) {
  try {
    // Wrap metadata extraction in a 10s timeout
    return await runWithTimeout(async () => {
      const loadingTask = pdfjsLib.getDocument({
        url: pdfUrl,
        standardFontDataUrl: standardFontDataUrl,
        verbosity: 0 // Suppress warnings
      });
      const pdf = await loadingTask.promise;
      
      const metadata = await pdf.getMetadata();
      const outline = await pdf.getOutline(); // This can sometimes be heavy
      
      const pageMetadata = [];
      const pagesToSample = Math.min(3, pdf.numPages); // Reduced sample size for speed
      
      for (let i = 1; i <= pagesToSample; i++) {
        const page = await pdf.getPage(i);
        const viewport = page.getViewport({ scale: 1.0 });
        pageMetadata.push({
          pageNumber: i,
          width: viewport.width,
          height: viewport.height,
        });
        // Cleanup page resources immediately
        page.cleanup(); 
      }
      return {
        numPages: pdf.numPages,
        metadata: metadata.info || {},
        outline: outline,
        pageMetadata: pageMetadata
      };
    }, 10000, "Metadata Extraction");
  } catch (err) {
    console.warn(`Skipping metadata for ${path.basename(pdfUrl)}: ${err.message}`);
    return null;
  }
};

// ------------------------------------------------------------------
// FIXED: Robust Image Extraction
// ------------------------------------------------------------------
doc2txt.extractImagesWithMetadataFromPdf = async function(pdfUrl) {
  try {
    // 60 Second timeout for ALL images in a file
    return await runWithTimeout(async () => {
      const loadingTask = pdfjsLib.getDocument({
        url: pdfUrl,
        standardFontDataUrl: standardFontDataUrl,
        verbosity: 0
      });
      const pdf = await loadingTask.promise;
      const imageData = [];
      
      // Limit to first 20 pages to prevent OOM on massive reports
      const maxPages = Math.min(pdf.numPages, 20); 

      for (let i = 1; i <= maxPages; i++) {
        try {
          const page = await pdf.getPage(i);
          const operatorList = await page.getOperatorList();
          
          // CRITICAL: If a page is too complex (e.g., >5000 ops = vector graph), skip it
          if (operatorList.fnArray.length > 5000) {
            console.warn(`Skipping images on page ${i} (Too complex/Vector Graph)`);
            page.cleanup();
            continue;
          }

          const processingPromises = [];

          for (let j = 0; j < operatorList.fnArray.length; j++) {
            const fn = operatorList.fnArray[j];
            const args = operatorList.argsArray[j];
            
            if (
              fn === pdfjsLib.OPS.paintJpegXObject ||
              fn === pdfjsLib.OPS.paintImageXObject ||
              fn === pdfjsLib.OPS.paintImageXObjectMask
            ) {
              const imgName = args[0];
              
              // CRITICAL FIX: Ensure Promise ALWAYS resolves
              const promise = new Promise((resolve) => {
                try {
                  page.objs.get(imgName, async (image) => {
                    if (image && image.data) {
                      try {
                        let base64String;
                        if (image.kind === pdfjsLib.ImageKind.JPEG) {
                          const buffer = Buffer.from(image.data);
                          base64String = 'data:image/jpeg;base64,' + buffer.toString('base64');
                        } else {
                          base64String = await rgbaToPngBase64(image);
                        }
                        if (base64String) {
                          imageData.push({
                            pageNumber: i,
                            imageName: imgName,
                            base64: base64String,
                            width: image.width,
                            height: image.height
                          });
                        }
                      } catch (e) {
                         // ignore conversion error
                      }
                    }
                    resolve(); // Always resolve
                  });
                } catch (e) {
                  resolve(); // Resolve if objs.get fails synchronously
                }
              });
              processingPromises.push(promise);
            }
          }

          await Promise.all(processingPromises);
          page.cleanup(); // Free memory
        } catch (pageErr) {
          console.warn(`Error on page ${i}: ${pageErr.message}`);
        }
      }
      return imageData;
    }, 60000, "Image Extraction"); // 60s hard limit

  } catch (err) {
    console.warn(`Image extraction timed out or failed: ${err.message}`);
    return [];
  }
};

doc2txt.readTextFromDoc = async function (filePath) {
  try {
    const extractor = new WordExtractor();
    return (await extractor.extract(filePath)).getBody();
  } catch (err) {
    return "";
  }
};

doc2txt.readTextFromDocx = async function (filePath) {
  try {
    const zip = await JSZip.loadAsync(await fs.readFile(filePath));
    const xmlFile = await zip.file('word/document.xml').async('string');
    const result = await parseStringPromise(xmlFile);
    const body = result?.['w:document']?.['w:body'];
    if (!body || !Array.isArray(body)) return "";

    let text = body[0]['w:p']?.map(paragraph => {
      return paragraph['w:r']?.map(run => {
        const content = run['w:t'];
        if (Array.isArray(content)) {
          return content.map(textRun => textRun._).join('');
        } else if (content && content._) {
          return content._;
        } else {
          return '';
        }
      }).filter(Boolean).join(' ');
    }).filter(Boolean).join('\n') || '';
    return text.trim();
  } catch (err) {
    return "";
  }
};

// ------------------------------------------------------------------
// FIXED: Main PDF Reader
// ------------------------------------------------------------------
doc2txt.readTextFromPdf = async function (filePath, options = {}) {
  // Use a default object to populate results
  const result = { text: "", images: [], metadata: null };

  // 1. Extract Text (Max 30 seconds)
  try {
    result.text = await runWithTimeout(
      readPdfText({ url: filePath, verbosity: 0 }), 
      30000, 
      "Text Extraction"
    );
    console.log(result.text)
  } catch (e) {
    console.error(`❌ Text extraction ERROR for ${path.basename(filePath)}:`, e);
  }

  // 2. Extract Images (Max 60 seconds) - Run sequentially, not parallel to save RAM
  try {
    if (options.includeImageMetadata) {
        result.images = await doc2txt.extractImagesWithMetadataFromPdf(filePath);
    } 
    // You can add the fallback for extractImagesFromPdf here if you still use it
  } catch (e) {
    console.error(`  ⚠️ Image extraction failed/timeout: ${path.basename(filePath)}`);
  }

  // 3. Extract Metadata (Max 5 seconds)
  try {
    result.metadata = await doc2txt.extractPdfMetadata(filePath);
  } catch (e) {
    // Ignore metadata failure
  }

  return result;
};

doc2txt.extractTextFromFile = async function (filePath) {
  const extension = path.extname(filePath).toLowerCase();
  try {
    switch (extension) {
      case '.txt': case '.py': case '.js': case '.sh': case '.md': case '.csv': case '.json':
        return { text: await this.readTextFromTxt(filePath), images: [], metadata: null };
      case '.doc':
        return { text: await this.readTextFromDoc(filePath), images: [], metadata: null };
      case '.docx':
        return { text: await this.readTextFromDocx(filePath), images: [], metadata: null };
      case '.pdf':
        // NOTE: We now default to including metadata/images. 
        // Pass options if you want to toggle this behavior.
        return await this.readTextFromPdf(filePath, { includeImageMetadata: true });
      default:
        return { text: "", images: [], metadata: null };
    }
  } catch (err) {
    console.error(`Critical error on file ${path.basename(filePath)}: ${err.message}`);
    return { text: "", images: [], metadata: null };
  }
};

export default doc2txt;
