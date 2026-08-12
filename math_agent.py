"""Retrieval-augmented math tutor with source citations and a SymPy scratchpad."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sympy as sp
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from ingestion_pipline import COLLECTION_NAME
load_dotenv()

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Role: You are an Advanced Math AI Agent specializing in problems from high-school
through university level. Your expertise includes Algebra (linear and abstract),
Calculus (limits, derivatives, integrals, and differential equations), and Probability
& Statistics.

Internal reasoning framework:
Use a programmer's mindset and verify calculations with SymPy, NumPy, or SciPy when
appropriate. Keep implementation details internal; the student wants a mathematical
solution, not the Python implementation.

Strict response pipeline — follow this order in every answer:
1. **Analysis** — summarize the problem, list the given conditions and constraints,
   state the target, and classify the mathematical sub-field.
2. **Method** — describe the mathematical method using normal mathematical language.
3. **Result** — provide the verified exact result. Never guess numerical values.
4. **Step-by-step explanation** — give an elegant, pedagogical solution. Explain
   transformations such as factoring, integration by parts, substitution, Bayes'
   theorem, or matrix decomposition.

Core principles:
- Zero hallucination: verify computations internally with symbolic or numerical logic,
  but do not expose the verification code in the student-facing answer.
- Ambiguity handling: identify missing data or unclear wording, state a reasonable
  assumption, and show how the code would change under another assumption.
- Answer in the same language as the user's question.
- Textbook excerpts are supporting context, not a hard restriction. Never claim that a
  textbook says something unless it appears in CONTEXT.
- Follow-up resolution: if the newest question is a contextual request such as
  "explain it again", "explain again", "say that again", "giải thích lại",
  "hãy giải thích lại", or "giải thích rõ hơn", resolve “it” from the most recent
  Tutor answer in RECENT CONVERSATION. Re-explain that same result immediately in
  simpler language, with the key steps expanded. Do not ask the student to repeat the
  topic unless RECENT CONVERSATION is empty or genuinely contains no prior solution.

Formatting rules:
- Use LaTeX for mathematics: inline `$...$`, display `$$...$$` on separate lines.
- Never show Python code, imports, library names, function calls, variable declarations,
  or implementation terms such as `diff`, `symbols`, `sympy`, `numpy`, or `scipy`.
- Do not show expressions in programming notation such as `x**2`; write them as
  readable LaTeX, for example $x^2$.
- Do not use square brackets as equation delimiters.
- If a graph is supplied, connect the explanation and check-yourself questions to its
  intercepts, slope, turning points, asymptotes, area, or other visible features.

CONTEXT:
{context}

RECENT CONVERSATION:
{conversation_history}

GRAPH:
{graph_context}

SCRATCHPAD (optional symbolic check; treat as a check, not as a source):
{scratchpad}"""),
    ("human", "{question}"),
])


def format_math_for_streamlit(text: str) -> str:
    """Normalize common model LaTeX styles to Streamlit's Markdown math syntax."""
    # Hide implementation details if the model accidentally exposes its internal
    # verification code in the final response.
    text = re.sub(
        r"(?im)^\s*(?:from\s+sympy.*|import\s+(?:sympy|numpy|scipy).*|(?:from\s+)?sympy\s+import.*|.*\b(?:symbols|diff|integrate|lambdify)\s*\(.*)$\n?",
        "",
        text,
    )
    text = re.sub(r"(?im)^\s*(?:derivative_[a-zA-Z_]+|[a-zA-Z_]+_f)\s*$\n?", "", text)

    # Convert common Python-style polynomial assignments into readable math.
    def convert_polynomial_assignment(match: re.Match[str]) -> str:
        name, expression = match.group(1), match.group(2).strip()
        expression = expression.replace("**", "^")
        expression = re.sub(r"(?<![\w)])\s*\*\s*(?=[a-zA-Z])", "", expression)
        expression = re.sub(r"\b([0-9]+)\s*\*\s*([a-zA-Z])", r"\1\2", expression)
        return f"$$\n{name}(x) = {expression}\n$$"

    text = re.sub(
        r"(?m)^\s*([a-zA-Z])\s*=\s*([0-9a-zA-Z_+*/().\-\s^]+)\s*$",
        convert_polynomial_assignment,
        text,
    )
    text = re.sub(r"(?m)^\s*\$\$\s*\$\$\s*$", "", text)

    # A model may accidentally wrap an explanation in ```java, ```text, or ```.
    # Those fences force Streamlit to render the contents as plain code.
    text = re.sub(r"(?m)^\s*```[A-Za-z0-9_+-]*\s*$", "", text)

    # Convert standard LaTeX delimiters first.
    text = re.sub(r"\\\[(.*?)\\\]", r"\n$$\1$$\n", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)

    # Models sometimes emit a display equation as [ ... ]. Only convert a whole
    # line when it contains clear mathematical LaTeX, avoiding Markdown links.
    def convert_bracket_equation(match: re.Match[str]) -> str:
        equation = match.group(1).strip()
        if "\\" in equation or "^" in equation or "_" in equation:
            return f"\n$$\n{equation}\n$$\n"
        return match.group(0)

    text = re.sub(r"(?m)^\s*\[\s*(.*?)\s*\]\s*$", convert_bracket_equation, text)

    # Convert common bare math lines, for example:
    # "At x = \frac{\pi}{2}, y = 0" -> "At $x = ...$, $y = 0$"
    def convert_value_line(match: re.Match[str]) -> str:
        prefix, expression = match.group(1), match.group(2).strip()
        expression = re.sub(r"\s*,\s*", r"$, $", expression)
        return f"{prefix}${expression}$"

    text = re.sub(
        r"(?m)^(\s*(?:[-*]\s*)?At\s+)(x\s*=.*)$",
        convert_value_line,
        text,
        flags=re.IGNORECASE,
    )

    # Wrap standalone LaTeX equations that the model omitted delimiters for,
    # such as ``\sin(0) = 0`` or ``\sin(\pi/2) = 1``.
    def convert_bare_math_line(match: re.Match[str]) -> str:
        line = match.group(1).strip()
        return f"\n$${line}$$\n"

    text = re.sub(
        r"(?m)^\s*((?:\\(?:sin|cos|tan|cot|sec|csc|pi|frac|sqrt|lim|infty)|[fy]\s*\([^\n]+\)|y\s*=)[^\n]*)\s*$",
        convert_bare_math_line,
        text,
    )

    # Also format short inline function expressions inside prose.
    text = re.sub(
        r"(?<![$\w])(y\s*=\s*\\(?:sin|cos|tan)\s*\([^\n)]*\))(?=[\s.,;:]|$)",
        r"$\1$",
        text,
    )
    # Handle complete trigonometric equalities before wrapping the function
    # alone, so ``sin(pi/2) = 1`` stays one readable expression.
    text = re.sub(
        r"(?<![$\\\w])(\\(?:sin|cos|tan)\s*(?:\\left)?\([^\n]*?(?:\\right)?\)\s*=\s*[-+]?\d+)(?![$\w])",
        r"$\1$",
        text,
    )

    text = re.sub(
        r"(?<![$\\\w])(\\(?:sin|cos|tan|cot|sec|csc)(?:\\left)?\([^\n,)]+(?:\\right)?\))(?=[\s.,;:]|$)",
        r"$\1$",
        text,
    )
    text = re.sub(
        r"(?<![$\w])(x\s*=\s*\\frac\{[^{}]+\}\{[^{}]+\})(?![$\w])",
        r"$\1$",
        text,
    )
    text = re.sub(
        r"(?<![$\\\w])(\\(?:sin|cos|tan)\s*(?:\\left)?\([^\n]*?(?:\\right)?\)\s*=\s*[-+]?\d+)(?![$\w])",
        r"$\1$",
        text,
    )
    text = re.sub(
        r"(?<![$\\\w{])((?:\d+\s*)?\\pi(?:\s*/\s*\d+)?)(?![\w$}])",
        r"$\1$",
        text,
    )
    text = re.sub(r"\$\$\s*\$\$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def symbolic_scratchpad(question: str) -> str:
    """Make a conservative symbolic check for common algebra/calculus prompts."""
    expression = re.search(r"(?:calculate|simplify|evaluate)\s+([0-9a-zA-Z+\-*/^(). ]+)", question, re.I)
    if not expression:
        return "No automatic symbolic check was run."
    raw = expression.group(1).strip().replace("^", "**")
    try:
        return f"SymPy simplified `{raw}` to `{sp.simplify(sp.sympify(raw))}`."
    except (sp.SympifyError, TypeError, ValueError):
        return "The expression was not safely parseable by the symbolic checker."


def extract_function(question: str) -> Optional[Tuple[str, str]]:
    """Extract a simple y=f(x) expression for an optional graph."""
    text = question.lower().replace("^", "**")
    if not re.search(r"\b(?:graph|plot|visuali[sz]e|draw|sketch)\b", text):
        return None
    named_functions = {
        "sine": "sin(x)", "sin": "sin(x)",
        "cosine": "cos(x)", "cos": "cos(x)",
        "tangent": "tan(x)", "tan": "tan(x)",
        "exponential": "exp(x)",
    }
    for name, expression in named_functions.items():
        if re.search(rf"\b{name}(?:\s+function)?\b", text):
            return "y", expression
    patterns = [
        r"(?:y|f\s*\(\s*x\s*\))\s*=\s*([0-9a-zx+\-*/().\s]+?)(?=\s+(?:and|with|where|for|please)\b|[?.!,;:]|$)",
        r"(?:graph|plot|visuali[sz]e)\s+(?:the\s+)?(?:function\s+)?([0-9a-zx+\-*/().\s]+?)(?=\s+(?:and|with|where|for|please)\b|[?.!,;:]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            expression = match.group(1).strip().rstrip("?.!,;:")
            if "x" in expression:
                return "y", expression
    return None


def create_graph(question: str) -> Optional[Tuple[Dict[str, List[float]], str]]:
    """Create lightweight graph data for a function explicitly in the question."""
    function = extract_function(question)
    if not function:
        return None
    _, expression = function
    try:
        import numpy as np

        x = sp.symbols("x")
        allowed = {"x": x, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                   "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs}
        formula = sp.sympify(expression, locals=allowed)
        if formula.has(sp.Symbol("y")) or formula.free_symbols - {x}:
            return None
        values = sp.lambdify(x, formula, modules=["numpy"])(np.linspace(-10, 10, 800))
        values = np.asarray(values, dtype=float)
        if values.ndim != 1:
            return None
        values[~np.isfinite(values)] = np.nan
        values = np.clip(values, -100, 100)
        x_values = np.linspace(-10, 10, 800)

        return {"x": x_values.tolist(), "y": values.tolist()}, expression
    except (ImportError, SyntaxError, TypeError, ValueError, NameError, sp.SympifyError):
        return None


class MathAgent:
    def __init__(self, persist_directory: str = "db/chroma_db") -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY in .env before starting Proof.")
        self.store = None
        if Path(persist_directory).exists():
            try:
                self.store = Chroma(
                    collection_name=COLLECTION_NAME,
                    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
                    persist_directory=persist_directory,
                )
            except Exception:
                # Cloud deployments may intentionally omit the large local index.
                # The tutor can still answer general math questions without RAG.
                self.store = None
        self.llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5.6"), temperature=0.1)

    @staticmethod
    def _format_history(messages: Optional[List[Dict[str, object]]]) -> str:
        """Keep enough recent turns for meaningful follow-up questions."""
        if not messages:
            return "No earlier conversation in this session."
        turns = []
        for message in messages[-8:]:
            role = "Student" if message.get("role") == "user" else "Tutor"
            content = str(message.get("content", "")).strip()
            if content:
                turns.append(f"{role}: {content}")
        history = "\n\n".join(turns)
        return history[-12000:] if history else "No earlier conversation in this session."

    def ask(
        self,
        question: str,
        history: Optional[List[Dict[str, object]]] = None,
        k: int = 5,
    ) -> Tuple[str, List[Dict[str, object]], Optional[Tuple[Dict[str, List[float]], str]]]:
        follow_up = bool(re.fullmatch(
            r"\s*(?:please\s+)?(?:explain|say)\s+(?:it\s+)?again\s*[.!?]*\s*|"
            r"\s*(?:hãy\s+)?giải\s*thích\s*(?:lại|rõ\s+hơn)\s*[.!?]*\s*",
            question,
            flags=re.IGNORECASE,
        ))
        if follow_up and history:
            previous_answer = next(
                (str(message.get("content", "")).strip()
                 for message in reversed(history)
                 if message.get("role") == "assistant" and message.get("content")),
                "",
            )
            if previous_answer:
                question = (
                    "Re-explain the previous tutor answer below in simpler language. "
                    "Keep the same topic and result, expand each key transformation, "
                    "and do not ask the student to provide the question again.\n\n"
                    f"Previous tutor answer:\n{previous_answer}"
                )
        graph = create_graph(question)
        visualization_requested = bool(re.search(r"\b(?:graph|plot|visuali[sz]e|draw|sketch)\b", question, re.I))
        if self.store is not None:
            ranked_docs = self.store.similarity_search_with_relevance_scores(question, k=k)
            docs = [doc for doc, score in ranked_docs if score >= 0.20]
        else:
            docs = []
        # Chroma can return weak matches when the library contains unrelated books.
        # An explicit message helps the model distinguish “no useful textbook context”
        # from a real excerpt without preventing a useful general math answer.
        context = (
            "\n\n---\n\n".join(doc.page_content for doc in docs)
            if docs
            else "No textbook index is available in this deployment. Answer using general mathematical knowledge and do not claim textbook citations."
        )
        answer = (PROMPT | self.llm).invoke({
            "question": question,
            "context": context,
            "conversation_history": self._format_history(history),
            "graph_context": (
                f"A graph of y = {graph[1]} will be shown below the answer."
                if graph else (
                    "The student requested a visualization but no function was identified. "
                    "Choose a clear example function and write it as f(x) = ... ."
                    if visualization_requested else "No graph is being shown."
                )
            ),
            "scratchpad": symbolic_scratchpad(question),
        }).content
        if graph is None and visualization_requested:
            graph = create_graph(str(answer))
        sources = []
        for doc in docs:
            source = Path(str(doc.metadata.get("source", "unknown"))).name
            page = doc.metadata.get("page")
            sources.append({"source": source, "page": page, "preview": doc.page_content[:180]})
        return str(answer), sources, graph

