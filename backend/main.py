from fastapi import FastAPI
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import torch
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from transformers.pipelines import pipeline

app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load lightweight model
model_name = "distilgpt2"  # <<< Optimized for memory usage
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set up HuggingFace pipeline - correct task for CausalLM
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=100)
llm = HuggingFacePipeline(pipeline=pipe)

# Load some sample documents
loader = TextLoader("../data/sample.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(texts, embeddings)

# QA Chain
qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

@app.post("/query")
async def query_rag(data: dict):
    question = data.get("question")
    result = qa_chain.invoke({"query": question})
    return {"answer": result["result"]}
