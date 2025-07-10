import bs4
from langchain_community.document_loaders import WebBaseLoader

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

## add metadata
total_documents = len(all_splits)
third = total_documents // 3

for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["section"] = "beginning"
    elif i < 2 * third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"


print( f"Metadata = {all_splits[0].metadata}" )



# Step 3: Create embeddings
#embedding = OpenAIEmbeddings()

from langchain.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 4: Create vector store from documents
from langchain.vectorstores import FAISS
import os

persist_path = "vector_store.vs"
if os.path.exists(persist_path):
    vector_store = FAISS.load_local(persist_path, embeddings=embedding, allow_dangerous_deserialization=True)
else:
    vector_store = FAISS.from_documents(all_splits, embedding)
    vector_store.save_local("vector_store.vs")

document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])


from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# Define the RAG prompt locally
rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Context: {context}
Question: {question}
Answer:"""
)

# Example usage
prompt_text = rag_prompt.format(
    context="The capital of France is Paris.",
    question="What is the capital of France?"
)


example_message = rag_prompt.invoke(
    {"context": "(context goes here)", "question": "(question goes here)"}
).to_messages()

assert len(example_message) == 1
print(f"prompt message : {example_message[0].content}")


from langchain_core.documents import Document
from typing_extensions import List, TypedDict

from typing import AsyncGenerator

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

from typing import Literal
from typing_extensions import Annotated

class Search(TypedDict):
    """Search query."""
    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]
    

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    
def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}
    
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = rag_prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence([analyze_query,retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()

# result = graph.invoke({"question": "What is Task Decomposition?"})

# print(f'Context: {result["context"]}\n\n')
# print(f'Answer: {result["answer"]}')

# for step in graph.stream({"question": "What is Task Decomposition?"}, stream_mode="updates"):
#     print(f"{step}\n\n----------------\n")
    
for message, metadata in graph.stream({"question": "What is Task Decomposition?"}, stream_mode="messages"):
    print(message.content, end="|")