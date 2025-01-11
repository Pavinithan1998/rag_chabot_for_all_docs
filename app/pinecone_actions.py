import os
import pinecone
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_HOST = os.getenv("PINECONE_HOST")
pinecone_index_name = os.getenv("PINECONE_INDEX")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


#create docs folder if it doesn't exist
if not os.path.exists("docs"):
    os.makedirs("docs")

def clear_index():
    """Empties the Pinecone index by deleting all vectors, but keeps the index itself."""
    try:
        index = pinecone.Index(api_key=PINECONE_API_KEY, host=PINECONE_HOST)
        index.delete(delete_all=True)  # Deletes all vectors but keeps the index
        print("Index cleared successfully!")
        return True
    except Exception as e:
        print(f"Error clearing index: {e}")
        return False
    
def ingest_documents():
    """Ingest documents into local vector store."""
    try:
        print(f"Attempting to load file: ")
        chunk_size = 1000
        chunk_overlap = 200
        
        # loader = DirectoryLoader('./docs', glob="*.txt", loader_cls=TextLoader)
        loader = DirectoryLoader('./docs', glob="*.txt", loader_cls=lambda path: TextLoader(path, encoding="utf-8"))
        docs = loader.load()
        print(f"Loaded {len(docs)} documents.")

        text_splitter =  CharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = text_splitter.split_documents(docs)
        print(f"Split into {len(documents)} chunks.")

        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        vectors = [
            {
                "id": f"doc_{i}",
                "values": embeddings.embed_query(doc.page_content),
                "metadata": {
                    "source": doc.metadata.get("source", "unknown"),
                    "text": doc.page_content,  # Ensure text is added to metadata
                },
            }
            for i, doc in enumerate(documents)
        ]
        print(f"Generated embeddings for {len(vectors)} chunks.")
        index = pinecone.Index(api_key=PINECONE_API_KEY, host=PINECONE_HOST)

        index.upsert(vectors)
        print(f"Uploaded {len(vectors)} vectors to Pinecone index.")

        # after uploading the index, delete the all txt files in the docs folder
        for file in os.listdir("docs"):
            if file.endswith(".txt"):
                os.remove(f"./docs/{file}")
        print("All processed files have been deleted from the 'docs' folder.")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
    
from langchain_community.vectorstores import Pinecone as LangChainPinecone

def load_vectorstore():
    """Load vectorstore as a retriever using Pinecone."""
    # Wrap Pinecone index with LangChain
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    index = pinecone.Index(api_key=PINECONE_API_KEY, host=PINECONE_HOST)
    vectorstore = LangChainPinecone(index, embeddings.embed_query, "text")
    return vectorstore

