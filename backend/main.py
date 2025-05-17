from fastapi import FastAPI
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import torch
import textwrap
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load small LLM (example: TinyLlama)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#  for seq2seqlm model
# model_name = "google/flan-t5-small"    
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set up HuggingFace pipeline
pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=150)
llm = HuggingFacePipeline(pipeline=pipeline)

# Load some sample documents
loader = TextLoader("../data/sample.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
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
