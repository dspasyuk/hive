import path from 'path';
import JSZip from 'jszip';
import fs from 'fs/promises';
import { parseStringPromise } from 'xml2js';
import WordExtractor from 'word-extractor';
import { readPdfText } from 'pdf-text-reader';
import * as pdfjsLib from 'pdfjs-dist';

import { createCanvas, ImageData } from 'canvas';

async function rgbaToPngBase64(image) {
  const { width, height, data } = image;

  let rgba;
  if (data.length === width * height * 3) {
    // RGB → RGBA
    rgba = new Uint8ClampedArray(width * height * 4);
    for (let i = 0, j = 0; i < data.length; i += 3, j += 4) {
      rgba[j] = data[i];       // R
      rgba[j + 1] = data[i+1]; // G
      rgba[j + 2] = data[i+2]; // B
      rgba[j + 3] = 255;       // A
    }
  } else if (data.length === width * height) {
    // Grayscale → RGBA
    rgba = new Uint8ClampedArray(width * height * 4);
    for (let i = 0, j = 0; i < data.length; i++, j += 4) {
      const val = data[i];
      rgba[j] = rgba[j+1] = rgba[j+2] = val;
      rgba[j+3] = 255;
    }
  } else {
    // Already RGBA
    rgba = new Uint8ClampedArray(data);
  }

  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  const imgData = new ImageData(rgba, width, height);
  ctx.putImageData(imgData, 0, 0);

  return canvas.toDataURL('image/png');
}

// Set the workerSrc property to the correct worker path
pdfjsLib.GlobalWorkerOptions.workerSrc = 'pdfjs-dist/build/pdf.worker.mjs';

const doc2txt = {};

// Utility to suppress console warnings temporarily
const suppressWarnings = (callback) => {
  const originalWarn = console.warn;
  console.warn = (...args) => {
    const msg = args.join(' ');
    // Suppress specific PDF.js warnings
    if (
      msg.includes('fetchStandardFontData') ||
      msg.includes('getPathGenerator') ||
      msg.includes('standardFontDataUrl') ||
      msg.includes('baseUrl') ||
      msg.includes('decodeScan') ||
      msg.includes('unexpected MCU data') ||
      msg.includes('marker is:')
    ) {
      // Silently ignore these warnings
      return;
    }
    // Allow other warnings through
    originalWarn.apply(console, args);
  };
  
  const result = callback();
  
  // Restore console.warn after a short delay to catch async warnings
  if (result && typeof result.then === 'function') {
    return result.finally(() => {
      setTimeout(() => {
        console.warn = originalWarn;
      }, 100);
    });
  } else {
    console.warn = originalWarn;
    return result;
  }
};

// Read text from a .txt file
doc2txt.readTextFromTxt = async function (filePath) {
  try {
    return await fs.readFile(filePath, 'utf8');
  } catch (err) {
    console.error("Error reading .txt file:", err.message);
    return "";
  }
};

// Extract metadata from PDF (new function)
doc2txt.extractPdfMetadata = async function(pdfUrl) {
  return suppressWarnings(async () => {
    try {
      const loadingTask = pdfjsLib.getDocument(pdfUrl);
      const pdf = await loadingTask.promise;
      
      const metadata = await pdf.getMetadata();
      const outline = await pdf.getOutline();
      
      // Get page-level metadata for first few pages
      const pageMetadata = [];
      const pagesToSample = Math.min(5, pdf.numPages); // Sample first 5 pages
      
      for (let i = 1; i <= pagesToSample; i++) {
        const page = await pdf.getPage(i);
        const viewport = page.getViewport({ scale: 1.0 });
        
        pageMetadata.push({
          pageNumber: i,
          width: viewport.width,
          height: viewport.height,
          rotation: viewport.rotation
        });
      }
      return {
        numPages: pdf.numPages,
        fingerprints: pdf.fingerprints,
        metadata: {
          title: metadata.info?.Title || null,
          author: metadata.info?.Author || null,
          subject: metadata.info?.Subject || null,
          keywords: metadata.info?.Keywords || null,
          creator: metadata.info?.Creator || null,
          producer: metadata.info?.Producer || null,
          creationDate: metadata.info?.CreationDate || null,
          modificationDate: metadata.info?.ModDate || null,
          pdfVersion: metadata.info?.PDFFormatVersion || null,
          encrypted: metadata.info?.IsAcroFormPresent || false,
          linearized: metadata.info?.IsLinearized || false,
          pageLayout: metadata.info?.PageLayout || null,
          pageMode: metadata.info?.PageMode || null,
        },
        outline: outline,
        pageMetadata: pageMetadata,
        rawMetadata: metadata.metadata?.getAll ? metadata.metadata.getAll() : null
      };
    } catch (err) {
      console.error("Error extracting PDF metadata:", err.message);
      return null;
    }
  });
};

// Keep original function for backward compatibility
doc2txt.extractImagesFromPdf = async function(pdfUrl) {
  return suppressWarnings(async () => {
    try {
      const loadingTask = pdfjsLib.getDocument(pdfUrl);
      const pdf = await loadingTask.promise;
      const extractedImages = [];

      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const operatorList = await page.getOperatorList();
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
            
            const promise = new Promise((resolve) => {
              page.objs.get(imgName, async (image) => {
                if (image && image.data) {
                  let base64String;
                  try {
                    if (image.kind === pdfjsLib.ImageKind.JPEG) {
                      const buffer = Buffer.from(image.data);
                      base64String = 'data:image/jpeg;base64,' + buffer.toString('base64');
                    } else {
                      base64String = await rgbaToPngBase64(image);
                    }
                    
                    if (base64String) {
                      extractedImages.push({
                        base64: base64String,
                        width: image.width,
                        height: image.height
                      });
                    }
                  } catch (e) {
                    console.error(`Error processing image ${imgName}:`, e.message);
                  }
                }
                resolve();
              });
            });
            processingPromises.push(promise);
          }
        }

        await Promise.all(processingPromises);
      }

      return extractedImages;
    } catch (err) {
      console.error("Error extracting images from PDF:", err.message);
      return [];
    }
  });
};

// New function with enhanced metadata
doc2txt.extractImagesWithMetadataFromPdf = async function(pdfUrl) {
  return suppressWarnings(async () => {
    try {
      const loadingTask = pdfjsLib.getDocument(pdfUrl);
      const pdf = await loadingTask.promise;
      const imageData = [];

      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const operatorList = await page.getOperatorList();
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
            const promise = new Promise((resolve) => {
              page.objs.get(imgName, async (image) => {
                if (image && image.data) {
                  let base64String;
                  try {
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
                        height: image.height,
                        kind: image.kind
                      });
                    }
                  } catch (e) {
                    console.error(`Error processing image ${imgName}:`, e.message);
                  }
                }
                resolve();
              });
            });
            processingPromises.push(promise);
          }
        }

        await Promise.all(processingPromises);
      }

      return imageData;
    } catch (err) {
      console.error("Error extracting images with metadata from PDF:", err.message);
      return [];
    }
  });
};

// Read text from a .doc file
doc2txt.readTextFromDoc = async function (filePath) {
  try {
    const extractor = new WordExtractor();
    return (await extractor.extract(filePath)).getBody();
  } catch (err) {
    console.error("Error reading .doc file:", err.message);
    return "";
  }
};

// Read text from a .docx file
doc2txt.readTextFromDocx = async function (filePath) {
  try {
    const zip = await JSZip.loadAsync(await fs.readFile(filePath));
    const xmlFile = await zip.file('word/document.xml').async('string');

    const result = await parseStringPromise(xmlFile);
    const body = result?.['w:document']?.['w:body'];
    if (!body || !Array.isArray(body)) {
      console.warn("Invalid DOCX structure: 'w:body' element is missing or malformed.");
      return "";
    }

    // Extract text by handling multiple nested levels in the XML structure
    let text = body[0]['w:p']?.map(paragraph => {
      return paragraph['w:r']?.map(run => {
        // Extract text if present and ensure it's a string
        const content = run['w:t'];
        if (Array.isArray(content)) {
          return content.map(textRun => textRun._).join(''); // Join multiple text runs
        } else if (content && content._) {
          return content._; // Single text run
        } else {
          return ''; // No text content
        }
      }).filter(Boolean).join(' '); // Remove empty strings and join text in run
    }).filter(Boolean).join('\n') || ''; // Remove empty strings in paragraph and join text
    return text.trim();
  } catch (err) {
    console.error("Error reading DOCX file:", err.message);
    return "";
  }
};

// Enhanced PDF reading with metadata and warning suppression
doc2txt.readTextFromPdf = async function (filePath, options = {}) {
  return suppressWarnings(async () => {
    try {
      // Extract text from the PDF (suppress warnings from readPdfText too)
      let text = "";
      try {
        text = await readPdfText({ url: filePath, verbosity: 0 });
      } catch (textErr) {
        console.error("Error extracting text from PDF:", textErr.message);
        // Continue with empty text, try to get images and metadata
      }
      
      // Extract images from the same PDF (choose format based on options)
      let images = [];
      try {
        images = options.includeImageMetadata 
          ? await this.extractImagesWithMetadataFromPdf(filePath)
          : await this.extractImagesFromPdf(filePath);
      } catch (imgErr) {
        console.error("Error extracting images from PDF:", imgErr.message);
        // Continue with empty images
      }

      // Extract metadata from the PDF
      let metadata = null;
      try {
        metadata = await this.extractPdfMetadata(filePath);
      } catch (metaErr) {
        console.error("Error extracting metadata from PDF:", metaErr.message);
        // Continue with null metadata
      }

      // Return an object containing text, images, and metadata
      return {
        text: text,
        images: images,
        metadata: metadata
      };
    } catch (err) {
      console.error("Error reading PDF file:", err.message);
      return { text: "", images: [], metadata: null };
    }
  });
};

// Extract text from a file based on its extension (updated to include metadata)
doc2txt.extractTextFromFile = async function (filePath) {
  const extension = path.extname(filePath).toLowerCase();
  try {
    switch (extension) {
      case '.txt':
      case '.py':
      case '.js':
      case '.sh':
      case '.md':
      case '.csv':
      case '.json':
        return {
          text: await this.readTextFromTxt(filePath),
          images: [],
          metadata: null
        };
      case '.doc':
        return {
          text: await this.readTextFromDoc(filePath),
          images: [],
          metadata: null
        };
      case '.docx':
        return {
          text: await this.readTextFromDocx(filePath),
          images: [],
          metadata: null
        };
      case '.pdf':
        // The readTextFromPdf function now returns text, images, and metadata
        // with warnings suppressed
        return await this.readTextFromPdf(filePath);
      default:
        console.warn(`Unsupported file format: ${extension}`);
        return { text: "", images: [], metadata: null };
    }
  } catch (err) {
    console.error("Error extracting text from file:", err.message);
    return { text: "", images: [], metadata: null };
  }
};

export default doc2txt;