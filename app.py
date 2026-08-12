"""MathMind-style Streamlit interface for the Proof RAG tutor."""

import streamlit as st
import sympy as sp
import pandas as pd

from math_agent import MathAgent, format_math_for_streamlit

st.set_page_config(page_title="MathMind · AI math tutor", page_icon="∑", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');
:root { --ink:#0b1c30; --muted:#59677a; --navy:#001e40; --aqua:#89f5e7; --pale:#eff4ff; --line:#c3c6d1; --canvas:#f8f9ff; }
.stApp { background:var(--canvas); color:var(--ink); font-family:Inter,sans-serif; }
[data-testid="stSidebar"] { background:#eff4ff; border-right:1px solid #c3c6d1; }
[data-testid="stSidebar"] > div:first-child { padding:1.4rem 1rem; }
[data-testid="stSidebar"] .stButton button { border:0; text-align:left; background:transparent; color:#59677a; border-radius:10px; padding:.75rem .9rem; }
[data-testid="stSidebar"] .stButton button:hover { background:#d3e4fe; color:#001e40; }
.brand { font:700 1.5rem 'Hanken Grotesk',sans-serif; color:#001e40; margin:.1rem 0 .15rem; }
.brand-sub { color:#59677a; font-size:.78rem; margin-bottom:1.5rem; }
.new-problem button { background:#001e40 !important; color:#fff !important; text-align:center !important; font-weight:600; }
.page-header { display:flex; justify-content:space-between; align-items:center; padding:.7rem 0 1.2rem; border-bottom:1px solid rgba(195,198,209,.45); margin-bottom:1.4rem; }
.page-title { font:700 2rem 'Hanken Grotesk',sans-serif; color:#001e40; }
.eyebrow { color:#006a61; text-transform:uppercase; letter-spacing:.12em; font-size:.68rem; font-weight:700; }
.topic { background:#eff4ff; border:1px solid rgba(195,198,209,.5); padding:1rem 1.1rem; border-radius:14px; margin-bottom:1.4rem; }
.topic-title { color:#001e40; font-weight:700; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
.topic-copy { color:#59677a; margin-top:.2rem; }
.assistant-card, .history-card, .formula-card { background:#fff; border:1px solid rgba(195,198,209,.55); border-radius:16px; padding:1.25rem 1.4rem; box-shadow:0 4px 12px rgba(0,0,0,.04); margin-bottom:1rem; }
.assistant-label { color:#59677a; font-size:.8rem; font-weight:600; margin-bottom:.8rem; }
.assistant-dot { display:inline-flex; width:28px; height:28px; align-items:center; justify-content:center; background:#003366; color:#799dd6; border-radius:50%; margin-right:.45rem; }
.formula { background:#f8f9ff; border:1px solid #d3e4fe; color:#001e40; border-radius:10px; padding:1rem; font:500 1rem 'JetBrains Mono',monospace; overflow:auto; }
.section-title { font:700 1.4rem 'Hanken Grotesk',sans-serif; color:#001e40; margin:1rem 0; }
.stChatInput { bottom:1rem; }
div[data-testid="stChatMessage"] { background:transparent; }
</style>
""", unsafe_allow_html=True)


def go_to(view: str) -> None:
    st.session_state.view = view


if "view" not in st.session_state:
    st.session_state.view = "chat"
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown('<div class="brand">MathMind</div><div class="brand-sub">Algorithmic Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="new-problem">', unsafe_allow_html=True)
    if st.button("＋  New Problem", use_container_width=True):
        st.session_state.messages = []
        go_to("chat")
    st.markdown('</div>', unsafe_allow_html=True)
    st.write("")
    for label, view in [("◌  Chat", "chat"), ("⌗  Solver", "solver"), ("◷  History", "history"), ("▦  Formula Library", "library")]:
        if st.button(label, key=f"nav_{view}", use_container_width=True):
            go_to(view)
    st.divider()
    st.caption("Textbooks are managed by the developer in the docs folder.")


def render_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="page-header"><div><div class="eyebrow">MathMind tutor</div><div class="page-title">{title}</div></div><div style="color:#59677a">∑</div></div><div class="topic"><div class="topic-title">Current workspace</div><div class="topic-copy">{subtitle}</div></div>', unsafe_allow_html=True)


def render_sources(sources: list[dict[str, object]]) -> None:
    if sources:
        with st.expander("Textbook passages used"):
            for source in sources:
                page = f", p. {source['page']}" if source.get("page") else ""
                st.caption(f"{source['source']}{page} — {source['preview']}…")


def render_chat() -> None:
    render_header("Solve, explore, understand.", "Ask a question grounded in your developer-managed mathematics textbooks.")
    if not st.session_state.messages:
        st.markdown('<div class="assistant-card"><div class="assistant-label"><span class="assistant-dot">∑</span>MathMind Assistant</div><div>Welcome! Ask me to explain a concept, solve a problem, or visualize a function.</div></div>', unsafe_allow_html=True)
        st.caption("Try one of these:")
        suggestions = ["Explain the chain rule with an example", "Graph y = x^2 and explain the turning point", "Solve 2(x - 3) + 4x = 5x + 7"]
        cols = st.columns(3)
        for col, suggestion in zip(cols, suggestions):
            if col.button(suggestion, key=f"suggest_{suggestion}", use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(format_math_for_streamlit(message["content"]))
            if message.get("graph"):
                st.caption(f"Graph of y = {message['expression']}")
                st.line_chart(pd.DataFrame(message["graph"]).set_index("x"), y="y", height=300)
            render_sources(message.get("sources", []))

    question = st.chat_input("Ask a math question or type an equation…")
    question = question or st.session_state.pop("pending_question", None)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        try:
            if "agent" not in st.session_state:
                st.session_state.agent = MathAgent()
            with st.spinner("MathMind is working through it…"):
                answer, sources, graph = st.session_state.agent.ask(question)
            graph_data = graph[0] if graph else None
            expression = graph[1] if graph else None
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources, "graph": graph_data, "expression": expression})
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def render_solver() -> None:
    render_header("Equation solver", "Work through algebra step by step and verify the result.")
    equation = st.text_input("Equation", placeholder="2(x - 3) + 4x = 5x + 7")
    if st.button("Solve equation", type="primary") and equation:
        try:
            left, right = equation.split("=")
            x = sp.symbols("x")
            solution = sp.solve(sp.sympify(left) - sp.sympify(right), x)
            st.markdown('<div class="assistant-card"><div class="assistant-label">Solution</div>', unsafe_allow_html=True)
            st.markdown(format_math_for_streamlit(f"$$ {sp.latex(sp.Eq(sp.Symbol('x'), solution[0]))} $$" if solution else "No real solution found."))
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as exc:
            st.error(f"Could not parse that equation: {exc}")


def render_library() -> None:
    render_header("Formula library", "Quick references for the concepts in your math workspace.")
    formula_groups = {
        "Basic differentiation": [
            ("Constant", r"\frac{d}{dx}(c)=0"),
            ("Identity", r"\frac{d}{dx}(x)=1"),
            ("Power rule", r"\frac{d}{dx}(x^n)=nx^{n-1}"),
            ("Exponential base e", r"\frac{d}{dx}(e^x)=e^x"),
            ("Exponential base a", r"\frac{d}{dx}(a^x)=a^x\ln(a)"),
        ],
        "Logarithmic functions": [
            ("Natural logarithm", r"\frac{d}{dx}(\ln x)=\frac{1}{x}"),
            ("Logarithm base a", r"\frac{d}{dx}(\log_a x)=\frac{1}{x\ln(a)}"),
        ],
        "Trigonometric functions": [
            ("Sine", r"\frac{d}{dx}(\sin x)=\cos x"),
            ("Cosine", r"\frac{d}{dx}(\cos x)=-\sin x"),
            ("Tangent", r"\frac{d}{dx}(\tan x)=\sec^2 x"),
            ("Cotangent", r"\frac{d}{dx}(\cot x)=-\csc^2 x"),
            ("Secant", r"\frac{d}{dx}(\sec x)=\sec x\tan x"),
            ("Cosecant", r"\frac{d}{dx}(\csc x)=-\csc x\cot x"),
        ],
        "Inverse trigonometric functions": [
            ("Inverse sine", r"\frac{d}{dx}(\sin^{-1}x)=\frac{1}{\sqrt{1-x^2}}"),
            ("Inverse cosine", r"\frac{d}{dx}(\cos^{-1}x)=-\frac{1}{\sqrt{1-x^2}}"),
            ("Inverse tangent", r"\frac{d}{dx}(\tan^{-1}x)=\frac{1}{1+x^2}"),
            ("Inverse cotangent", r"\frac{d}{dx}(\cot^{-1}x)=-\frac{1}{1+x^2}"),
            ("Inverse secant", r"\frac{d}{dx}(\sec^{-1}x)=\frac{1}{|x|\sqrt{x^2-1}}"),
            ("Inverse cosecant", r"\frac{d}{dx}(\csc^{-1}x)=-\frac{1}{|x|\sqrt{x^2-1}}"),
        ],
        "Hyperbolic functions": [
            ("Hyperbolic sine", r"\frac{d}{dx}(\sinh x)=\cosh x"),
            ("Hyperbolic cosine", r"\frac{d}{dx}(\cosh x)=\sinh x"),
            ("Hyperbolic tangent", r"\frac{d}{dx}(\tanh x)=\operatorname{sech}^2 x"),
            ("Hyperbolic cotangent", r"\frac{d}{dx}(\coth x)=-\operatorname{csch}^2 x"),
            ("Hyperbolic secant", r"\frac{d}{dx}(\operatorname{sech}x)=-\operatorname{sech}x\tanh x"),
            ("Hyperbolic cosecant", r"\frac{d}{dx}(\operatorname{csch}x)=-\operatorname{csch}x\coth x"),
        ],
        "Inverse hyperbolic functions": [
            ("Inverse hyperbolic sine", r"\frac{d}{dx}(\sinh^{-1}x)=\frac{1}{\sqrt{1+x^2}}"),
            ("Inverse hyperbolic cosine", r"\frac{d}{dx}(\cosh^{-1}x)=\frac{1}{\sqrt{x^2-1}}"),
            ("Inverse hyperbolic tangent", r"\frac{d}{dx}(\tanh^{-1}x)=\frac{1}{1-x^2}"),
            ("Inverse hyperbolic cotangent", r"\frac{d}{dx}(\coth^{-1}x)=\frac{1}{1-x^2}"),
            ("Inverse hyperbolic secant", r"\frac{d}{dx}(\operatorname{sech}^{-1}x)=-\frac{1}{|x|\sqrt{1-x^2}}"),
            ("Inverse hyperbolic cosecant", r"\frac{d}{dx}(\operatorname{csch}^{-1}x)=-\frac{1}{|x|\sqrt{1+x^2}}"),
        ],
        "Rules of differentiation": [
            ("Product rule", r"\frac{d}{dx}[f(x)g(x)]=f'(x)g(x)+f(x)g'(x)"),
            ("Quotient rule", r"\frac{d}{dx}\left[\frac{f(x)}{g(x)}\right]=\frac{f'(x)g(x)-f(x)g'(x)}{[g(x)]^2}"),
            ("Chain rule", r"\frac{d}{dx}f(g(x))=f'(g(x))g'(x)"),
        ],
    }

    for group_name, formulas in formula_groups.items():
        st.markdown(f'<div class="section-title">{group_name}</div>', unsafe_allow_html=True)
        columns = st.columns(2)
        for index, (title, formula) in enumerate(formulas):
            with columns[index % 2]:
                with st.container(border=True):
                    st.caption(title)
                    st.latex(formula)


def render_history() -> None:
    render_header("Study history", "Your recent questions and worked explanations.")
    assistant_messages = [m for m in st.session_state.messages if m["role"] == "user"]
    if not assistant_messages:
        st.info("Your solved questions will appear here.")
    for index, message in enumerate(reversed(assistant_messages), 1):
        st.markdown(f'<div class="history-card"><div class="eyebrow">Question {index}</div><div style="font-weight:600;margin-top:.35rem">{message["content"]}</div></div>', unsafe_allow_html=True)


if st.session_state.view == "chat":
    render_chat()
elif st.session_state.view == "solver":
    render_solver()
elif st.session_state.view == "library":
    render_library()
else:
    render_history()
