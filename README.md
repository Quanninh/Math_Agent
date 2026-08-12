# MathMind — AI mathematics textbook tutor

MathMind is a local RAG agent for asking questions about calculus, algebra, and other mathematics textbooks. It reads `.pdf` and `.txt` files from `docs/`, stores semantic chunks in Chroma, retrieves the most relevant passages, and asks an OpenAI model to explain the answer with citations. A small SymPy scratchpad performs conservative checks for simple arithmetic/algebra expressions. The Streamlit interface uses the MathMind study workspace design with Chat, Solver, History, and Formula Library views.

## Run it

1. Create the Conda environment: `conda env create -f environment.yml`
2. Activate it: `conda activate proof`
3. Put your key in `.env`: `OPENAI_API_KEY=...`
4. Add textbooks to `docs/`.
5. Build the index: `python ingestion_pipline.py`
6. Start the app: `streamlit run app.py`

In VS Code, select the Conda interpreter with `Cmd+Shift+P` → `Python: Select Interpreter` → `proof` so Pylance uses the same environment.

Textbooks are managed by the developer in the `docs/` folder. After changing the library, rebuild the index from the sidebar or by running `python ingestion_pipline.py`. The local Chroma database is stored under `db/chroma_db/`.

The Conda environment installs the available foundations through Conda and the LangChain packages through its bundled pip step, because those LangChain builds are not available from the configured `osx-arm64` Conda channels. The `requirements.txt` file is retained for fully pip-based environments.

## Example questions

- “Explain the chain rule and show a worked example.”
- “What is the geometric meaning of a derivative?”
- “Compare completing the square with the quadratic formula.”
- “Calculate (3 + 5)^2 and explain each step.”

The agent is intentionally grounded in the indexed books. If the books do not contain enough information, it should say so instead of filling gaps from memory.
