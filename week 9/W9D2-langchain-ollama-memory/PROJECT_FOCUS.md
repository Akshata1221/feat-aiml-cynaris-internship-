# LangChain + Ollama Conversational Research Assistant

Create a LangChain prompt/model/output-parser chain powered by a local Ollama model, with explicit conversation history passed between turns.

## Run
Install Ollama and pull a model (default `llama3.1`), then:
```bash
pip install -r requirements.txt
ollama pull llama3.1
python examples/chat.py
```
Conversation history is held in the running Python process; it is not persistent across restarts.
