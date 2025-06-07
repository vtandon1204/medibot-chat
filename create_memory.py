# 1. Load raw PDF file

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader # type: ignore
import glob

DATA_PATH = 'data/'
def load_pdf(data):
    """Load a PDF file and return its content."""
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf(data = DATA_PATH)
print(f"Total pages loaded as documents: {len(documents)}")

pdf_files = glob.glob(f"{DATA_PATH}/*.pdf")
print(f"Total PDF files: {len(pdf_files)}")


# 2. Create chunks of text

from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore

def create_chunks(documents, chunk_size=500, chunk_overlap=50 ):
    """Split documents into chunks of text."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

text_chunks = create_chunks(documents)

print(f"Total chunks created: {len(text_chunks)}")

# 3. Create Vector embeddings of the chunks

from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
import torch

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"})
    
    return embedding_model

embedding_model = get_embedding_model()


# 4. Store the embeddings in a vector database (Faiss)

from langchain_community.vectorstores import FAISS # type: ignore

DB_PATH = 'vectorstore/db_faiss'
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_PATH)

