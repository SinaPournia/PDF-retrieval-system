---

# 📚 PDF Retrieval System with ColQwen2 and Vespa

![PDF Retrieval System Banner](https://cdn-uploads.huggingface.co/production/uploads/60f2e021adf471cbdf8bb660/La8vRJ_dtobqs6WQGKTzB.png)

## 🌟 Overview

This advanced PDF retrieval system combines **ColQwen2** (an enhanced version of ColPali) and **Vespa** to efficiently process, store, and retrieve PDF content using both textual and visual features.

## 📋 Prerequisites

Ensure you have the following tools installed before proceeding:

- **Vespa CLI**  
  Install via Homebrew:
  ```bash
  brew install vespa-cli
  ```
  [More details](https://docs.vespa.ai/en/vespa-cli.html)

- **Poetry**  
  Poetry is used to manage project dependencies. Install it with:
  ```bash
  curl -sSL https://install.python-poetry.org | python3 -
  ```

- **PDF Tools**  
  The project uses `pypdf` for text extraction and `pdf2image` for converting PDF pages to images, both of which are added via Poetry. However, `pdf2image` requires **Poppler**, a separate system tool. Install Poppler with the following commands:
  
  - On **macOS** (via Homebrew):
    ```bash
    brew install poppler
    ```

  - On **Ubuntu/Debian**:
    ```bash
    sudo apt-get install poppler-utils
    ```

  - On **Windows**:
    - Download Poppler from [here](http://blog.alivate.com.au/poppler-windows/).
    - Add the Poppler `bin` folder to your system's PATH environment variable.

## 🏃‍♂️ Getting Started

Once you have the prerequisites installed, follow these steps to set up and run the PDF Retrieval System:

1. **Install Project Dependencies**  
   Install the required dependencies for the project using Poetry:
   ```bash
   poetry install
   ```

2. **Start Docker Containers**  
   Navigate to the project folder and use Docker Compose to start the local Vespa container:
   ```bash
   docker compose up
   ```

3. **Create and Deploy Vespa Application**  
   Run `create_vespa_app.py` to automatically configure the Vespa application schema and deploy it using Vespa CLI.

4. **Generate and Upload Embeddings**  
   Execute `create_and_upload_embeddings.py` to generate embeddings from PDFs and upload them to the Vespa container.

5. **Retrieve and Generate Report**  
   Run `retrive_and_generate_report.py` to query the Vespa application and retrieve results. This will generate HTML files to visualize the retrieved results.

---

## 🔍 Process Flow

1. **Text Extraction**  
   Extract text from PDFs using the `pypdf` library.

2. **Image Creation**  
   Convert PDF pages to images with `pdf2image`.

3. **Embedding Generation**  
   Generate embeddings for each PDF page using ColQwen2.

4. **Binary Quantization**  
   Quantize embeddings to binary format, reducing their size by 32x (128-bit binary vectors).

5. **Data Preparation**  
   Organize extracted text, images, embeddings, and metadata into JSON objects formatted for Vespa.

6. **Vespa Schema Configuration and Deployment**  
   The Vespa schema is configured as follows:
   ```yaml
   fields:
     id: string
     url: string
     title: string
     page_number: int
     image: base64
     text: string
     embedding: tensor<int8>(patch{}, v[16])
   ```

7. **Data Storage**  
   Store the prepared data in Vespa via its API.

8. **Query Processing and Retrieval**  
   Implement text-based retrieval using BM25, and re-rank results using embeddings with nearest-neighbor search (MaxSim).

9. **Response Generation**  
   Display ranked results as HTML files, combining visual and textual content from PDFs.

---

## 🛠️ Key Components

| Component   | Description                                                 |
|-------------|-------------------------------------------------------------|
| ColQwen2    | Advanced embedding generator based on ColPali                |
| Vespa       | Vector database for efficient storage and retrieval          |
| HNSW Index  | Fast nearest-neighbor search on binary embeddings            |
| BM25        | Text-based initial retrieval                                 |
| MaxSim      | Re-ranking based on embedding similarity                     |
| Poetry      | Dependency management and packaging tool                     |

## ✨ Features

- Efficient storage with binary quantization for embeddings
- Fast retrieval using HNSW index and Hamming distance
- Combined ranking pipeline (textual and visual features)
- Scalable to large PDF collections
- Visual HTML representation of query results
- Easy dependency management using Poetry

## 🚀 Future Improvements

- [ ] Integrate LLM for answering queries from retrieved images
- [ ] Add multi-language support
- [ ] Expand metadata and filtering options
- [ ] Enhance HTML output with interactive elements

---