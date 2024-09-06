## Hive.js vector datbase 

Hive.js is a simple vector-based database system that provides functionality for inserting, storing, and querying high-dimensional vectors, with features such as transformer-based vector extraction and similarity-based search.  
It's built on top of Node.js. Database is stored in a json file.

## Features

- **Vector database:** Store and retrieve vectors efficiently using JSON-based storage.
- **Cosine Similarity Search:** Find the most similar vectors to a query using cosine similarity.
- **Transformer Integration:** Use transformer-based models to extract feature vectors from text.
- **File Support:** Read and process text files to insert into the vector database.
- **Token Management:** Supports token-based slicing of text for efficient vectorization.

## Installation

First, clone the repository and install the dependencies.

```bash
git clone https://github.com/dspasyuk/hive.git
cd hive
npm install
```

