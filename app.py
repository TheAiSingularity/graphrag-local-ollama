"""GraphRAG Local Ollama — Gradio Web UI.

Launch with:
    pip install -r requirements-ui.txt
    python app.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr
import yaml

ROOT = Path(__file__).parent
SETTINGS_FILE = ROOT / "settings.yaml"
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _load_settings() -> dict:
    if SETTINGS_FILE.exists():
        with SETTINGS_FILE.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_settings(data: dict) -> None:
    with SETTINGS_FILE.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def _run_subprocess(cmd: list[str]) -> tuple[str, str]:
    """Run a subprocess and return (stdout, stderr)."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    return result.stdout, result.stderr


def _stream_subprocess(cmd: list[str]):
    """Generator that yields lines from a subprocess in real time."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        bufsize=1,
    )
    for line in iter(proc.stdout.readline, ""):
        yield line
    proc.wait()
    yield f"\n[Exit code: {proc.returncode}]\n"


# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 — Indexing
# ──────────────────────────────────────────────────────────────────────────────


def upload_files(files) -> str:
    if not files:
        return "No files uploaded."
    names = []
    for f in files:
        dest = INPUT_DIR / Path(f.name).name
        Path(f.name).rename(dest) if Path(f.name).exists() else open(dest, "wb").write(open(f.name, "rb").read())
        names.append(dest.name)
    return f"Uploaded {len(names)} file(s): {', '.join(names)}"


def run_indexing() -> gr.update:
    cmd = [sys.executable, "-m", "graphrag.index", "--root", str(ROOT)]
    log_lines: list[str] = []

    def _work():
        for line in _stream_subprocess(cmd):
            log_lines.append(line)

    t = threading.Thread(target=_work, daemon=True)
    t.start()

    while t.is_alive() or log_lines:
        if log_lines:
            yield "".join(log_lines)
        else:
            yield "Running…"
        t.join(timeout=1)

    yield "".join(log_lines) + "\n✅ Indexing complete (or see errors above)."


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 — Query
# ──────────────────────────────────────────────────────────────────────────────


def run_query(question: str, method: str) -> str:
    if not question.strip():
        return "Please enter a question."
    method_flag = method.lower()
    cmd = [
        sys.executable, "-m", "graphrag.query",
        "--root", str(ROOT),
        "--method", method_flag,
        question,
    ]
    stdout, stderr = _run_subprocess(cmd)
    if stderr and not stdout:
        return f"Error:\n{stderr}"
    return stdout or stderr


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3 — Graph Visualizer
# ──────────────────────────────────────────────────────────────────────────────


def load_graph() -> tuple[str, str]:
    """Find the most recent GraphML snapshot and render it as an interactive HTML."""
    snapshots = sorted(OUTPUT_DIR.rglob("*.graphml"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not snapshots:
        return "", "No GraphML snapshot found. Run indexing first."

    graphml_path = snapshots[0]

    try:
        import networkx as nx
        from pyvis.network import Network

        G = nx.read_graphml(str(graphml_path))
        net = Network(height="600px", width="100%", notebook=False)
        net.from_nx(G)
        html_path = ROOT / "output" / "_graph_preview.html"
        net.save_graph(str(html_path))

        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        return html, f"Loaded: {graphml_path.name} ({len(G.nodes)} nodes, {len(G.edges)} edges)"
    except ImportError:
        return "", "pyvis is not installed. Run: pip install pyvis"
    except Exception as e:
        return "", f"Error loading graph: {e}"


# ──────────────────────────────────────────────────────────────────────────────
# Tab 4 — Settings
# ──────────────────────────────────────────────────────────────────────────────


def load_current_settings() -> tuple[str, str, int, int, bool]:
    s = _load_settings()
    llm_model = s.get("llm", {}).get("model", "mistral")
    embed_model = s.get("embeddings", {}).get("llm", {}).get("model", "nomic-embed-text")
    chunk_size = s.get("chunks", {}).get("size", 300)
    chunk_overlap = s.get("chunks", {}).get("overlap", 100)
    lazy = s.get("lazy_graph_rag", False)
    return llm_model, embed_model, chunk_size, chunk_overlap, lazy


def save_settings(llm_model: str, embed_model: str, chunk_size: int, chunk_overlap: int, lazy: bool) -> str:
    s = _load_settings()
    s.setdefault("llm", {})["model"] = llm_model
    s.setdefault("embeddings", {}).setdefault("llm", {})["model"] = embed_model
    s.setdefault("chunks", {})["size"] = int(chunk_size)
    s["chunks"]["overlap"] = int(chunk_overlap)
    s["lazy_graph_rag"] = lazy
    _save_settings(s)
    return "✅ Settings saved to settings.yaml"


# ──────────────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="GraphRAG Local Ollama", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🕸️ GraphRAG Local Ollama\nKnowledge graph RAG powered by local Ollama models.")

    with gr.Tab("📂 Index"):
        gr.Markdown("Upload text documents and build the knowledge graph index.")
        file_upload = gr.File(label="Upload documents (.txt)", file_count="multiple", file_types=[".txt", ".csv"])
        upload_btn = gr.Button("Upload Files")
        upload_status = gr.Textbox(label="Upload status", interactive=False)
        upload_btn.click(upload_files, inputs=file_upload, outputs=upload_status)

        gr.Markdown("---")
        index_btn = gr.Button("▶ Run Indexing", variant="primary")
        index_log = gr.Textbox(label="Indexing log", lines=20, max_lines=40, interactive=False)
        index_btn.click(run_indexing, inputs=[], outputs=index_log)

    with gr.Tab("🔍 Query"):
        gr.Markdown("Ask questions about your indexed documents.")
        question_box = gr.Textbox(label="Question", placeholder="What are the main themes in the documents?", lines=2)
        method_radio = gr.Radio(
            choices=["Global", "Local", "Drift", "Basic", "Lazy"],
            value="Global",
            label="Search method",
            info="Global: broad synthesis | Local: entity-focused | Drift: iterative graph reasoning | Basic: fast vector search | Lazy: no pre-computed reports needed",
        )
        query_btn = gr.Button("Ask", variant="primary")
        answer_box = gr.Textbox(label="Answer", lines=10, interactive=False)
        query_btn.click(run_query, inputs=[question_box, method_radio], outputs=answer_box)

    with gr.Tab("🗺️ Graph"):
        gr.Markdown("Visualize the knowledge graph from the most recent indexing run.")
        load_graph_btn = gr.Button("Load Graph")
        graph_status = gr.Textbox(label="Status", interactive=False)
        graph_html = gr.HTML(label="Graph")
        load_graph_btn.click(load_graph, inputs=[], outputs=[graph_html, graph_status])

    with gr.Tab("⚙️ Settings"):
        gr.Markdown("Edit key settings. Changes are written to `settings.yaml`.")
        with gr.Row():
            llm_model_input = gr.Textbox(label="LLM Model", value="mistral")
            embed_model_input = gr.Textbox(label="Embedding Model", value="nomic-embed-text")
        with gr.Row():
            chunk_size_input = gr.Number(label="Chunk size (tokens)", value=300, precision=0)
            chunk_overlap_input = gr.Number(label="Chunk overlap (tokens)", value=100, precision=0)
        lazy_toggle = gr.Checkbox(label="Enable LazyGraphRAG mode (skip community summaries at index time)", value=False)
        settings_save_btn = gr.Button("Save Settings", variant="primary")
        settings_status = gr.Textbox(label="Status", interactive=False)

        demo.load(
            load_current_settings,
            inputs=[],
            outputs=[llm_model_input, embed_model_input, chunk_size_input, chunk_overlap_input, lazy_toggle],
        )
        settings_save_btn.click(
            save_settings,
            inputs=[llm_model_input, embed_model_input, chunk_size_input, chunk_overlap_input, lazy_toggle],
            outputs=settings_status,
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
