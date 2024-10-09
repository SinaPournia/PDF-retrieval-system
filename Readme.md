# 📚 PDF Retrieval System with ColQwen2 and Vespa

![PDF Retrieval System Banner](https://cdn-uploads.huggingface.co/production/uploads/60f2e021adf471cbdf8bb660/La8vRJ_dtobqs6WQGKTzB.png)

## 🌟 Overview

This advanced PDF retrieval solution leverages the power of **ColQwen2** (an improved version of ColPali) and **Vespa** to efficiently process, store, and retrieve PDF content based on both textual and visual features.

## 🏃‍♂️ Getting Started

Follow these steps to set up and run the PDF Retrieval System:

1. **Install Dependencies**
   - Install the project dependencies using Poetry:
     ```
     poetry install
     ```
   - Activate the virtual environment:
     ```
     poetry shell
     ```

2. **Generate Vespa Application Folder**
   - Run `colqwen_create_app` to generate the Vespa application folder

3. **Deploy the Application**
   - Use Vespa CLI to set the target and deploy the generated app to a local Vespa container:
     ```
     vespa config set target local
     vespa deploy
     ```
   For more information, visit: https://docs.vespa.ai/en/vespa-quick-start.html

4. **Embed and Upload Data**
   - Execute `colqwen_upload_data` to generate embeddings and upload data to the local Vespa container

5. **Query the System**
   - Run `colqwen_query` to perform queries and retrieve results from the local address
   - This step generates HTML files to display the query results visually

For more detailed information on each step, refer to the documentation or contact the system administrator.

## 🔍 Process Flow

1. **Text Extraction** 📄
   - Extract text from PDFs using `pypdf`

2. **Image Creation** 🖼️
   - Convert PDF pages to images with `pdf2image`

3. **Embedding Generation** 🧠
   - Generate embeddings for each PDF page using ColQwen2

4. **Binary Quantization** 💾
   - Convert 128-dimensional embeddings into binary quantized form
   - Reduces size by 32x (128-bit binary vectors)

5. **Data Preparation** 🗃️
   - Organize text, images, embeddings, and metadata into JSON objects
   - Prepare "vespa_feed" in Vespa JSON format

6. **Vespa Schema Configuration and Deployment** 📐
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


7. **Data Storage** 💽
   - Store prepared data in Vespa via API

8. **Query Processing and Retrieval** 🔎
   - Text-based retrieval with embedding re-ranking (BM25)
   - Embedding-based retrieval (nearest neighbor + MaxSim)

9. **Response Generation** 📊
   - Rank and display retrieved PDF pages with visual and textual content
   - Generate HTML files to visualize query results

## 🛠️ Key Components

| Component | Description |
|-----------|-------------|
| ColQwen2 | Improved version of ColPali for embedding generation |
| Vespa | Vector database for efficient storage and retrieval |
| HNSW Index | Fast nearest neighbor search on binary embeddings |
| BM25 | Initial text-based retrieval |
| MaxSim | Re-ranking based on embedding similarity |
| Poetry | Dependency management and packaging tool |

## ✨ Features

- 📉 Efficient embedding storage via binary quantization
- 🚀 Fast retrieval using HNSW index and Hamming distance
- 🔀 Flexible ranking pipeline (textual + visual features)
- 📈 Scalable to large PDF collections
- 🖥️ Visual representation of query results through generated HTML files
- 📦 Easy dependency management with Poetry

## 🚀 Future Improvements

- [ ] Add LLM layer for querying retrieved images
- [ ] Implement multi-language support
- [ ] Expand metadata and filtering options
- [ ] Enhance HTML output with interactive features
- [ ] Streamline deployment process with containerization