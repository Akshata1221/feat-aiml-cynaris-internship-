\# Cynaris AI/ML Internship — Week 4 to Week 6



\## Project Overview



This repository contains my Artificial Intelligence and Machine Learning internship projects completed as part of the Cynaris Solutions AI/ML training program.



The projects focus on machine learning model evaluation, regularisation, model deployment, local large language models (LLMs), vector databases, semantic search, LangChain, AI agents, and Retrieval-Augmented Generation (RAG).



Each project demonstrates practical implementation using Python and modern AI/ML tools.



\## Technologies Used



\- \*\*Programming Language:\*\* Python

\- \*\*Machine Learning:\*\* Scikit-learn, NumPy, Pandas

\- \*\*Model Evaluation:\*\* Precision, Recall, F1-score, ROC-AUC

\- \*\*Experiment Tracking:\*\* MLflow

\- \*\*Model Deployment:\*\* FastAPI, Uvicorn

\- \*\*Local LLMs:\*\* Ollama, Llama 3.2, Qwen 2.5

\- \*\*Vector Database:\*\* ChromaDB

\- \*\*LLM Frameworks:\*\* LangChain, LangGraph

\- \*\*Additional Tools:\*\* Git, GitHub, VS Code, pytest, joblib



\---



\## Week 4 — Machine Learning Evaluation \& Deployment



Week 4 focuses on evaluating machine learning models, improving their performance, saving trained models, and deploying them through APIs.



| Day | Project | Description |

|---|---|---|

| W4D1 | Model Evaluation Metrics | Evaluate models using accuracy, precision, recall, F1-score, ROC-AUC, and cross-validation. |

| W4D2 | Bias-Variance Tradeoff \& Regularisation | Explore L1/L2 regularisation, learning curves, GridSearchCV, and RandomizedSearchCV. |

| W4D3 | Model Serialisation | Save and load machine learning models using joblib and pickle. |

| W4D4 | FastAPI Model Serving | Create REST API endpoints to serve machine learning predictions. |

| W4D5 | Sentiment Classifier Capstone | Compare Logistic Regression and Random Forest using text classification and deploy a sentiment prediction API. |



\### Week 4 Learning Outcomes



\- Understanding model evaluation metrics

\- Identifying overfitting and underfitting

\- Hyperparameter tuning and regularisation

\- Saving and loading trained models

\- Building machine learning prediction APIs

\- Comparing classification algorithms



\---



\## Week 5 — Local LLMs, Prompt Engineering \& Vector Databases



Week 5 introduces locally hosted large language models, prompt engineering, embeddings, and semantic search.



| Day | Project | Description |

|---|---|---|

| W5D1 | Running LLMs Locally with Ollama | Run Llama 3.2 and Qwen 2.5 locally and compare their responses. |

| W5D2 | Prompt Engineering \& System Prompts | Design custom system prompts and evaluate LLM responses. |

| W5D3 | ChromaDB Vector Store | Store document embeddings and retrieve relevant information using similarity search. |

| W5D4 | Semantic Search with ChromaDB | Implement cosine similarity, metadata filtering, and PDF chunk retrieval. |

| W5D5 | Local Q\&A Bot | Build a local question-answering application using Ollama and ChromaDB. |



\### Week 5 Learning Outcomes



\- Running LLMs without cloud APIs

\- Designing effective system prompts

\- Comparing responses from different LLMs

\- Generating and storing vector embeddings

\- Performing semantic similarity searches

\- Developing document-based question-answering systems



\---



\## Week 6 — LangChain, Agents \& Retrieval-Augmented Generation



Week 6 focuses on building intelligent applications using LangChain, conversation memory, AI agents, and RAG pipelines.



| Day | Project | Description |

|---|---|---|

| W6D1 | LangChain Fundamentals | Build prompt chains using Ollama and output parsers. |

| W6D2 | LangChain Memory | Maintain conversation history across multiple interactions. |

| W6D3 | LangChain Tools \& Agents | Build an agent with calculator and simulated web search tools. |

| W6D4 | RAG Pipeline | Integrate LangChain, ChromaDB, and Ollama for document retrieval and question answering. |

| W6D5 | Document Chatbot | Develop a document chatbot with conversation history and contextual retrieval. |



\### Week 6 Learning Outcomes



\- Building reusable LLM chains

\- Managing conversation history

\- Creating tool-enabled AI agents

\- Understanding the ReAct agent pattern

\- Implementing Retrieval-Augmented Generation

\- Developing document-based conversational AI applications



\---



\## Installation \& Setup



\### Prerequisites



\- Python 3.10 or later, with a version supported by the selected dependencies

\- Git

\- Visual Studio Code

\- Ollama (required for local LLM projects)



\### Clone Repository



```bash

git clone https://github.com/Akshata1221/feat-aiml-cynaris-internship-.git

cd feat-aiml-cynaris-internship-

```



\### Create a Virtual Environment



Navigate into the required project folder and run:



```bash

py -m venv .venv

.venv\\Scripts\\activate

```



\### Install Dependencies



```bash

python -m pip install -r requirements.txt

```



\### Run Automated Tests



```bash

python -m pytest -v

```



\### Set Up Ollama



Download Ollama from:



https://ollama.com/download



Pull the required models:



```bash

ollama pull llama3.2:3b

ollama pull qwen2.5:3b

ollama pull nomic-embed-text

```



Run the project-specific Python scripts according to the instructions in each project's README.



\---



\## Project Structure



```text

Cynaris-AIML-Internship/

│

├── week 4/

│   ├── W4D1-Model-Evaluation/

│   ├── W4D2-Bias-Variance-Regularisation/

│   ├── W4D3-Model-Serialisation/

│   ├── W4D4-FastAPI-Model-Serving/

│   └── W4D5-Sentiment-Classifier-Capstone/

│

├── week 5/

│   ├── W5D1-Ollama-Local-LLM/

│   ├── W5D2-Prompt-Engineering-Ollama/

│   ├── W5D3-ChromaDB-Ollama-RAG/

│   ├── W5D4-Semantic-Search-ChromaDB/

│   └── W5D5-Local-QA-Bot/

│

├── week 6/

│   ├── W6D1-LangChain-Chains-Agent/

│   ├── W6D2-LangChain-Memory-Agent/

│   ├── W6D3-LangChain-Tools-Agents/

│   ├── W6D4-LangChain-ChromaDB-RAG/

│   └── W6D5-Document-Chatbot-LangChain/

│

├── .gitignore

└── README.md

```



\## Testing \& Evaluation



The projects include automated tests and evaluation workflows where applicable.



Testing and evidence generation should be performed locally before submission. Results depend on the installed dependencies, available Ollama models, and runtime environment.



\## Git Workflow



The internship workflow uses weekly feature branches, descriptive commits, and pull requests.



Example branch:



`feat/aiml-W6-Akshata`



Each project should be reviewed and tested before committing.



\## Future Improvements



\- Improve model evaluation and experiment tracking

\- Expand document datasets

\- Add advanced RAG evaluation using Ragas

\- Improve AI agent tool integration

\- Deploy selected applications to cloud platforms

\- Add user interfaces for LLM-based applications



\---



\*This repository documents practical learning and project development in Machine Learning, Generative AI, LLM applications, and AI agent systems.\*

