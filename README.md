# GraphRAG Local Ollama

This repository is an adaptation of Microsoft's [GraphRAG](https://github.com/microsoft/graphrag), modified to support local models downloaded using Ollama, thus eliminating the dependency on costly OpenAPI models for inference.

## Installation and Setup

Follow these steps to set up and use this repository with local models for LLM and embeddings:

1. **Create and activate a new conda environment:**
    ```bash
    conda create -n graphrag-ollama-local python=3.10
    conda activate graphrag-ollama-local
    ```

2. **Install Ollama:**
    - Visit [Ollama's website](https://ollama.com/) for installation instructions.
    - After installation, run:
    ```bash
    pip install ollama
    ```

3. **Download the required models using Ollama:**
    ```bash
    ollama pull mistral
    ollama pull nomic-embed-text
    ```

4. **Clone the repository:**
    ```bash
    git clone https://github.com/TheAiSingularity/graphrag-local-ollama.git
    ```

5. **Install the necessary packages:**
    ```bash
    pip install -e .
    ```

6. **Navigate to the repository directory:**
    ```bash
    cd graphrag-local-ollama/
    ```

7. **Create the required input directory:**
    ```bash
    mkdir -p ./ragtest/input
    ```

8. **Initialize the indexing:**
    ```bash
    python3 -m graphrag.index --init --root ./ragtest
    ```

9. **Copy input files:**
    ```bash
    cp inputs/* ./ragtest/input
    ```

10. **Move the settings file:**
    ```bash
    mv settings.yaml ./ragtest
    ```

11. **Run the indexing:**
    ```bash
    python3 -m graphrag.index --root ./ragtest
    ```

12. **Run a query:**
    ```bash
    python3 -m graphrag.query --root ./ragtest --method global "explain machine learning"
    ```

## Citations

- Original GraphRAG repository by Microsoft: [GraphRAG](https://github.com/microsoft/graphrag)
- Ollama: [Ollama](https://ollama.com/)

---

By following the above steps, you can set up and use local models with GraphRAG, making the process more cost-effective and efficient.
