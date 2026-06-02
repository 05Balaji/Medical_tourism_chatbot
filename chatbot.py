from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import os
import requests
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma,FAISS
from langchain_ollama import ChatOllama
from langchain_google_genai import  ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from langgraph.graph import  StateGraph,END
from typing import TypedDict

from dotenv import load_dotenv

# import chromadb
# import ollama


urls = [
    "https://www.patient.nineremit.com/",
    "https://www.hospital.nineremit.com/"
]

web_text = ""
for url in urls:
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text,"html.parser")
        texts = soup.get_text()
        
        web_text += texts + "\n"
    except Exception as e:
        print(f"error in {url} : {e}")
        
reader = PdfReader("nineremit_data-compressed.pdf")
text = ""
for page in reader.pages:
    extracted = page.extract_text()
    
    if extracted:
        text += extracted + "\n"
        
combined_text = web_text + "\n" + text

        # chunks = combined_text.split("\n")
        # chunks = [chunk.strip() for chunk in chunks if chunk.strip() != ""]

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 500)
documents = splitter.create_documents([combined_text])

# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# client = chromadb.Client()
# collection = client.create_collection(name = "medical_tourism_data")

# vectorstore = Chroma.from_documents(documents = documents,
#                                     embedding = embedding_model,
#                                     persist_directory = "./chroma_db")

vectorstore = Chroma.from_documents(documents= documents,embedding=embedding_model,persist_directory = "./chroma_db")


retriever = vectorstore.as_retriever(search_kwargs = {"k":5})

# for i,chunk in enumerate(chunks):
#     embedding = embedding_model.encode(chunk).tolist()
#     collection.add(
#         ids = [str(i)],
#         embeddings = [embedding],
#         documents = [chunk]
#     )
# print("pdf data scored in vector data base")

# llm = ChatOllama(model = "phi3")

load_dotenv()

llm = ChatGroq(model = "llama-3.1-8b-instant",
               api_key =  os.getenv("Groq_API_KEY"))

prompt = PromptTemplate(
    input_variables = ["context","question"],
    template = """
            You are NineCure AI Assistant.
            Answer ONLY using the provided context.
            If the answer is not present in the context, say:
            'I could not find that information in the provided documents.'
            context:
            {context}

            Question:
            {question}

            Answer politely and professionally.
            """
)

class ChatState(TypedDict):
    question : str
    context : str
    answer : str
    
def retrieve_context(state : ChatState):
    question = state["question"]
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    
    return {
        "context" : context
    }
    
def generate_answer(state : ChatState):
    final_prompt = prompt.format(
        context = state["context"],
        question = state["question"]
    )
    try :
        response = llm.invoke(final_prompt)
        return {"answer":response.content}
    except Exception as e:
        return {"answer":f"error: {str(e)}"}

graph = StateGraph(ChatState)
graph.add_node("retrieve_context",retrieve_context)
graph.add_node("generate_answer",generate_answer)

graph.set_entry_point("retrieve_context")

graph.add_edge("retrieve_context","generate_answer")
graph.add_edge("generate_answer",END)

app_graph = graph.compile()


app = FastAPI()
# CORS FIX
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message : str


@app.get("/")
def home():
    return {
        "message" : "welcome to medical Tourism chatbot Api"
    }
    
@app.post("/chat")
def chat(request : ChatRequest):
    
    result = app_graph.invoke({
        "question" : request.message
    })
    
    return {
        "response" : result["answer"]
    }
    
