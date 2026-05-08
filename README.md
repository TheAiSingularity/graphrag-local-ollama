## 🤝 Contributing

**We welcome contributions from the community to help enhance GraphRAG Local Ollama! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details on how to get involved.**

Need support for llama integration.

# 🚀 GraphRAG Local Ollama - Knowledge Graph

Welcome to **GraphRAG Local Ollama**! This repository is an exciting adaptation of Microsoft's [GraphRAG](https://github.com/microsoft/graphrag), tailored to support local models downloaded using Ollama. Say goodbye to costly OpenAPI models and hello to efficient, cost-effective local inference using Ollama!

## 📄 Research Paper

For more details on the GraphRAG implementation, please refer to the [GraphRAG paper](https://arxiv.org/pdf/2404.16130).

**Paper Abstract**

The use of retrieval-augmented generation (RAG) to retrieve relevant information from an external knowledge source enables large language models (LLMs)to answer questions over private and/or previously unseen document collections.However, RAG fails on global questions directed at an entire text corpus, suchas “What are the main themes in the dataset?”, since this is inherently a queryfocused summarization (QFS) task, rather than an explicit retrieval task. PriorQFS methods, meanwhile, fail to scale to the quantities of text indexed by typicalRAG systems. To combine the strengths of these contrasting methods, we proposea Graph RAG approach to question answering over private text corpora that scaleswith both the generality of user questions and the quantity of source text to be indexed. Our approach uses an LLM to build a graph-based text index in two stages:first to derive an entity knowledge graph from the source documents, then to pregenerate community summaries for all groups of closely-related entities. Given aquestion, each community summary is used to generate a partial response, beforeall partial responses are again summarized in a final response to the user. For aclass of global sensemaking questions over datasets in the 1 million token range,we show that Graph RAG leads to substantial improvements over a na¨ıve RAGbaseline for both the comprehensiveness and diversity of generated answers. 

## 🌟 Features

- **Local Model Support:** Leverage local models with Ollama for LLM and embeddings.
- **Cost-Effective:** Eliminate dependency on costly OpenAPI models.
- **Easy Setup:** Simple and straightforward setup process.
- **Web UI:** Browser-based interface for indexing, querying, and graph visualization.
- **LazyGraphRAG Mode:** Skip community summarization at index time — ~99% faster indexing, on-demand summarization at query time.
- **Multilingual:** Full UTF-8 support with CJK-aware text chunking (Chinese, Japanese, Korean, Arabic, Cyrillic).
- **Docker:** One-command setup with `docker-compose up`.

---

## 🐳 Quick Start (Docker)

The fastest way to get started — no Python environment needed.

```bash
# 1. Copy the example environment file
cp .env.example .env

# 2. Start Ollama + GraphRAG
docker-compose up --build

# 3. Drop your .txt documents into ./input/
#    Indexing runs automatically on container start.

# 4. Run a query
docker-compose run graphrag python -m graphrag.query \
  --root /app --method global "What are the main themes?"
```

> **GPU support:** Add `deploy.resources.reservations.devices` to the `ollama` service in `docker-compose.yml` for GPU acceleration.

---

## 🖥️ Web UI

A browser-based interface for the full workflow (indexing → querying → visualization).

```bash
pip install -r requirements-ui.txt
python app.py
# Open http://localhost:7860
```

**Tabs:**
| Tab | What it does |
|-----|-------------|
| 📂 Index | Upload `.txt` files, run indexing, see live log output |
| 🔍 Query | Ask questions using Global / Local / Lazy search |
| 🗺️ Graph | Interactive knowledge-graph visualizer (requires pyvis) |
| ⚙️ Settings | Edit model names, chunk size, LazyGraphRAG toggle |

---

## ⚡ LazyGraphRAG Mode

Inspired by Microsoft's LazyGraphRAG, this mode **skips community summarization at index time** and generates summaries on-the-fly during queries. Result: indexing is ~99% faster.

**Enable in `settings.yaml`:**
```yaml
lazy_graph_rag: true
```

**Query with the lazy method:**
```bash
python -m graphrag.query --root ./ragtest --method lazy "What is machine learning?"
```

> **Trade-off:** First-query responses are slightly slower than standard global search because summaries are computed at query time. For large datasets this is still dramatically cheaper overall.

---

## 🌐 Non-English Text

UTF-8 is enforced throughout the pipeline. CJK (Chinese, Japanese, Korean), Arabic, and Cyrillic documents are supported:

- Text files are read with `encoding="utf-8"` (with graceful replacement for undecodable bytes).
- CSVs default to `utf-8` (was `latin-1` — fixed in this release).
- The text splitter automatically switches to **character-count chunking** for CJK-dominant text, avoiding BPE tokenization artifacts.

No configuration change needed — just drop your non-English `.txt` files in `input/` and run normally.

---

## 📦 Installation and Setup

Follow these steps to set up this repository and use GraphRag with local models provided by Ollama :


1. **Create and activate a new conda environment:  (please stick to the given python version 3.10 for no errors)**
    ```bash
    conda create -n graphrag-ollama-local python=3.10
    conda activate graphrag-ollama-local
    ```

2. **Install Ollama:**
    - Visit [Ollama's website](https://ollama.com/) for installation instructions.
    - Or, run:
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh #ollama for linux
    pip install ollama
    ```

3. **Download the required models using Ollama, we can choose from (mistral,gemma2, qwen2) for llm and any embedding model provided under Ollama:**
    ```bash
    ollama pull mistral  #llm
    ollama pull nomic-embed-text  #embedding
    ```

4. **Clone the repository:**
    ```bash
    git clone https://github.com/TheAiSingularity/graphrag-local-ollama.git
    ```

5. **Navigate to the repository directory:**
    ```bash
    cd graphrag-local-ollama/
    ```

6. **Install the graphrag package ** This is the most important step :**
    ```bash
    pip install -e .
    ```


7. **Create the required input directory: This is where the experiments data and results will be stored - ./ragtest**
    ```bash
    mkdir -p ./ragtest/input
    ```
    
8. **Copy sample data folder input/  to  ./ragtest. Input/ has the sample data to run the setup. You can add your own data here in .txt format.**
    ```bash
    cp input/* ./ragtest/input
    ```
    
9. **Initialize the ./ragtest folder to create the required files:**
    ```bash
    python -m graphrag.index --init --root ./ragtest
    ```

10. **Move the settings.yaml file, this is the main predefined config file configured with ollama local models :**
    ```bash
    cp settings.yaml ./ragtest
    ```

Users can experiment by changing the models. The llm model expects language models like llama3, mistral, phi3, etc., and the embedding model section expects embedding models like mxbai-embed-large, nomic-embed-text, etc., which are provided by Ollama. You can find the complete list of models provided by Ollama here https://ollama.com/library, which can be deployed locally. Both LLM and embeddings use the OpenAI-compatible endpoint `http://localhost:11434/v1`. No API key is required — the config defaults to `ollama` as a placeholder. 

![LLM Configuration](<Screenshot 2024-07-09 at 3.34.31 AM-1.png>)

![Embedding Configuration](<Screenshot 2024-07-09 at 3.36.28 AM.png>)

11. **Run the indexing, which creates a graph:**
    ```bash
    python -m graphrag.index --root ./ragtest
    ```

12. **Run a query (three methods available):**
    ```bash
    # Global search — broad synthesis across the whole corpus
    python -m graphrag.query --root ./ragtest --method global "What is machine learning?"

    # Local search — entity-focused, uses knowledge graph + text chunks
    python -m graphrag.query --root ./ragtest --method local "What is machine learning?"

    # Lazy search — no pre-computed community reports needed (use with lazy_graph_rag: true)
    python -m graphrag.query --root ./ragtest --method lazy "What is machine learning?"
    ```

**Graphs can be saved which further can be used for visualization by changing the graphml to "true" in the settings.yaml :**
    
    snapshots:
    graphml: true
    
**To visualize the generated graphml files, you can use : https://gephi.org/users/download/ or the script provided in the repo visualize-graphml.py :**

Pass the path to the .graphml file to the below line in visualize-graphml.py:

    graph = nx.read_graphml('output/20240708-161630/artifacts/summarized_graph.graphml') 

13. **Visualize .graphml :**

    ```bash
    python visualize-graphml.py
    ```



## Citations

- Original GraphRAG repository by Microsoft: [GraphRAG](https://github.com/microsoft/graphrag)
- Ollama: [Ollama](https://ollama.com/)

---

By following the above steps, you can set up and use local models with GraphRAG, making the process more cost-effective and efficient.
