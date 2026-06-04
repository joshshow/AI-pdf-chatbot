# AI-pdf-chatbot
Built an LLM powered Chatbot that used the Retrieval-Augmented Generation (RAG) architecture to answer questions from an uploaded PDF using semantic based searches.

Technologies Used
- Python
- PyPdf
- Langchain
- Hugging Face
- ChromaDB

How it works:
- The project has a five phase structure which involves (Document Injestion -> Text Chunking -> Embedding -> Vector Database -> LLM Powered Replies)
- The chosen pdf is uploaded to the system, a question asked, the system uses the RAG architecture to retrieve answers from the document with minimal hallucination.
- The tasks for the system were divided into two separate files to aid in debugging and code readability.
