import ast, operator
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
def safe_calculate(expression):
    def ev(node):
        if isinstance(node, ast.Expression): return ev(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float): return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS: return OPS[type(node.op)](ev(node.left), ev(node.right))
        raise ValueError("Only simple arithmetic is allowed")
    if len(expression) > 80: raise ValueError("Expression too long")
    return ev(ast.parse(expression, mode="eval"))

@tool
def calculator(expression: str) -> str:
    """Calculate simple arithmetic expressions, e.g. 5*(3+2)."""
    return str(safe_calculate(expression))

@tool
def web_search_stub(query: str) -> str:
    """Return a clearly labeled simulated search result; this tool has no internet access."""
    return f"STUB ONLY: No live web search performed for: {query}"

def build_agent():
    return create_react_agent(ChatOllama(model="llama3.2:3b", temperature=0), [calculator, web_search_stub])
