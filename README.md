## What is LangChain? 

LangChain is a framework for building applications powered by LLMs (Large Language Models).
It provides tools to connect:

- ✅ LLMs
- ✅ Prompts
- ✅ Memory
- ✅ Tools & APIs
- ✅ Document loaders
- ✅ Vector databases
- ✅ Agents & chains

In simple words:

👉 LangChain = “LLM + Tools + Workflow”
It helps developers build AI apps faster, cleaner, and modular.

##  Why LangChain?

Because raw OpenAI API gives you only a single text completion.
LangChain helps you build entire AI workflows:

- ✅ Chatbots
- ✅ Retrieval systems
- ✅ RAG systems
- ✅ Agents with tools
- ✅ Multi-step pipelines
- ✅ Custom workflows (planning, reasoning)

##  Key Features of LangChain (Explained Simply)
### 1. Prompt Templates

Reusable prompts with variables.

###  2. Chains

Connect multiple steps in a pipeline (e.g., summarize → translate → answer).

###  3. Agents + Tools

Let the LLM “decide” which tool to use:

- ✅ Google search
- ✅ Calculator
- ✅ Database
- ✅ API calls

### 4. Memory

Store conversation history.

### 5. Document Loading

PDF, DOCX, Web pages, YouTube transcripts, etc.

### 6. Vector DB Integration

For RAG (Retrieval-Augmented Generation):

- ✅ FAISS
- ✅ Pinecone
- ✅ Chroma
- ✅ Weaviate

### 7. LangSmith + [LangGraph](https://github.com/Manuhd/LangGraph/blob/main/README.md) Integration

LangChain + LangGraph = production-grade agent systems.

###  Practical LangChain Project

Here is a ready-to-run LangChain project:

- ✅ Conversation AI
- ✅ Document search (RAG)
- ✅ Tools
- ✅ Simple agent

#### Install
```
pip install langchain langchain-openai chromadb
pip install tiktoken
```
## 1️⃣ Basic LLM Call Using LangChain

```
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

response = llm.invoke("Explain blockchain in simple terms.")
print(response.content)
```

## 2️⃣ Prompt Template Example

```
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "Write a short message like a {style} about {topic}"
)

prompt = template.format(style="teacher", topic="AI")
print(prompt)
```

## 3️⃣ Chain Example

```
from langchain_core.runnables import RunnableSequence

chain = RunnableSequence([
    template,     # format prompt
    llm           # send to LLM
])

output = chain.invoke({"style": "engineer", "topic": "LangChain"})
print(output.content)
```

## 4️⃣ Memory Chatbot
```
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

memory = ConversationBufferMemory(return_messages=True)

from langchain.chains import ConversationChain

chat = ConversationChain(
    llm=llm,
    memory=memory
)

print(chat.run("Hello, who are you?"))
print(chat.run("What did I just ask you?"))
```
## 5️⃣ Simple Agent With Tool Use
```
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

agent = initialize_agent(
    tools=[multiply],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent.run("What is 12 multiplied by 8?")
```
## 6️⃣ RAG Example (Local Vector Store + PDF / Text Search)
✅ Create Vector Store
```
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

text = """
LangChain is a framework to build LLM applications.
It supports prompt templates, memory, tools, agents, and retrieval.
"""

splitter = RecursiveCharacterTextSplitter(chunk_size=200)
docs = splitter.split_text(text)

db = Chroma.from_texts(docs, embedding=OpenAIEmbeddings())
retriever = db.as_retriever()
```
### ✅ Ask a Question Using RAG
```
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

print(qa.run("What is LangChain?"))
```


“LangChain is a powerful framework that makes it easy to build AI applications using Large Language Models.
It provides ready-made components like prompts, chains, memory, agents, and retrieval systems, allowing LLMs to interact with data, tools, and workflows.
With LangChain, developers can create smart chatbots, RAG systems, document-search apps, and AI agents efficiently and in a structured way.”
