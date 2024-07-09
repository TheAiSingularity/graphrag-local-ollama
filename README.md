# 🚀 GraphRAG Local Ollama

Welcome to **GraphRAG Local Ollama**! This repository is an exciting adaptation of Microsoft's [GraphRAG](https://github.com/microsoft/graphrag), tailored to support local models downloaded using Ollama. Say goodbye to costly OpenAPI models and hello to efficient, cost-effective local inference using Ollama!

## 📄 Research Paper

For more details on the GraphRAG implementation, please refer to the [GraphRAG paper](https://arxiv.org/pdf/2404.16130).

## 🌟 Features

- **Local Model Support:** Leverage local models with Ollama for LLM and embeddings.
- **Cost-Effective:** Eliminate dependency on costly OpenAPI models.
- **Easy Setup:** Simple and straightforward setup process.

## 📦 Installation and Setup

Follow these steps to set up this repository and use GraphRag with local models provided by Ollama :


1. **Create and activate a new conda environment:**
    ```bash
    conda create -n graphrag-ollama-local python=3.10
    conda activate graphrag-ollama-local
    ```

2. **Install Ollama:**
    - Visit [Ollama's website](https://ollama.com/) for installation instructions.
    - Or, run:
    ```bash
    pip install ollama
    ```

3. **Download the required models using Ollama, we can choose any llm and embedding model provided under Ollama:**
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

6. **Install the graphrag package ***This is the most important step ***:**
    ```bash
    pip install -e .
    ```


7. **Create the required input directory: This is where the experiments data and results will be stored - ./ragtest**
    ```bash
    mkdir -p ./ragtest/input
    ```

8. **Initialize the ./ragtest folder to create the required files:**
    ```bash
    python -m graphrag.index --init --root ./ragtest
    ```

9. **Copy sample data folder input/  to  ./ragtest. Input/ has the sample data to run the setup. You can add your own data here in .txt format.**
    ```bash
    cp input/* ./ragtest/input
    ```
    **Export a dummy key as mentioned below or create a ./ragtest/.env file with GRAPHRAG_API_KEY=1234**
   ```bash
    export GRAPHRAG_API_KEY=1234
    ```

10. **Move the settings.yaml file, this is the main predefined config file configured with ollama local models :**
    ```bash
    mv settings.yaml ./ragtest
    ```

Users can experiment by changing the models. The llm model expects language models like llama3, mistral, phi3, etc., and the embedding model section expects embedding models like mxbai-embed-large, nomic-embed-text, etc., which are provided by Ollama. You can find the complete list of models provided by Ollama here https://ollama.com/library, which can be deployed locally. The default API base URLs are http://localhost:11434/v1 for LLM and http://localhost:11434/api for embeddings, hence they are added to the respective sections. 

![LLM Configuration](<Screenshot 2024-07-09 at 3.34.31 AM-1.png>)

![Embedding Configuration](<Screenshot 2024-07-09 at 3.36.28 AM.png>)

11. **Run the indexing, which creates a graph:**
    ```bash
    python -m graphrag.index --root ./ragtest
    ```

12. **Run a query: Only supports Global method** 
    ```bash
    python -m graphrag.query --data ./ragtest/output/20240709-024831/artifacts/ --method global "What is machine learning?"
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
