from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline

embedding_model = None
kollection = None
llm = None

def initialize_rag():

    global embedding_model
    global kollection
    global llm

    reader = PdfReader("sample.pdf")
    text = ""

    for karen in reader.pages:
        text += karen.extract_text()

    print("Document Injestion Successful")

    mastersplitter = RecursiveCharacterTextSplitter(
        chunk_size = 400,
        chunk_overlap = 80
    )

    chunks = mastersplitter.split_text(text)
    print("Text Chunking Successful")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedd = embedding_model.encode(chunks)
    print("Text Embedding Successful")

    klient = chromadb.Client()
    kollection = klient.create_collection(name="pdf-chatbot")
    for i, chunk in enumerate(chunks):
        kollection.add(
            ids = [str(i)],
            documents = [chunk],
            embeddings = [embedd[i].tolist()]
        )
    print("Vector Storage Successful")

    llm = pipeline("text2text-generation", model="google/flan-t5-base")
    print("RAG initialized Successfully")

def ask_pdf(question):

    que = embedding_model.encode(question)
    retrieved = kollection.query(
        query_embeddings = [que.tolist()],
        n_results = 3
    )

    context = "\n\n".join(retrieved["documents"][0])

    prompt = f""" Answer the question using the context below

    Question:
    {question}

    Context:
    {context}

    """
    answer = llm(prompt, max_length=256)
    ai_answer = answer[0]["generated_text"]

    return ai_answer
