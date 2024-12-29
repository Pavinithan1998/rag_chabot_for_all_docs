import os
import fitz  
import pinecone
from io import BytesIO
from docx import Document
from PyPDF2 import PdfReader
from pptx import Presentation
from langchain.vectorstores import Pinecone
from llm_actions import get_image_description
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
pinecone_index_name = os.getenv("PINECONE_INDEX")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

def ingest_document(text_content: str):
    """
    Ingest the document into the Pinecone vector store by splitting into chunks, vectorizing, and storing.
    """
    try:
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text_content)

        if pinecone_index_name not in pinecone.list_indexes():
            pinecone.create_index(pinecone_index_name, dimension=1536)  # 1536 is the default for OpenAI embeddings.
        index = Pinecone(index_name=pinecone_index_name, embedding=embeddings)

        index.add_texts(chunks)

        return "success"
    except Exception as e:
        return f"Failed to ingest document into Pinecone: {e}"
    
def process_pdf(file_bytes: bytes):
    """
    Process PDF file to extract text and image descriptions, integrating them into the text content.
    """
    text_content = ""
    pdf_document = fitz.open("pdf", file_bytes)
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text = page.get_text()
        text_content += f"Page {page_num + 1}:\n{text}\n"

        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            try:
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                
                description = get_image_description(image_bytes)
                
                text_content += f"\n[Image: {image_filename}]\nDescription: {description}\n"
            
            except Exception as e:
                text_content += f"\n[Image Extraction Failed: Page {page_num + 1}, Image {img_index + 1}]\nError: {e}\n"
    status = ingest_document(text_content)
    # print(text_content)
    return text_content

def process_docx(file_bytes: bytes):
    """Process DOCX file using fitz to extract text and image descriptions."""
    text_content = ""
    docx_pdf = fitz.open("docx", file_bytes)
    
    for page_num in range(docx_pdf.page_count):
        page = docx_pdf.load_page(page_num)
        text = page.get_text()
        text_content += f"Page {page_num + 1}:\n{text}\n"
        
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            try:
                base_image = docx_pdf.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                
                description = get_image_description(image_bytes)
                
                text_content += f"\n[Image: {image_filename}]\nDescription: {description}\n"
            except Exception as e:
                text_content += f"\n[Image Extraction Failed: Page {page_num + 1}, Image {img_index + 1}]\nError: {e}\n"
    
    status = ingest_document(text_content)
    # print(text_content)
    return text_content

def process_txt(file_bytes: bytes):
    """Process TXT file to extract plain text and store in Pinecone vector store."""
    try:
        text_content = file_bytes.decode("utf-8")
        
        status = ingest_document(text_content)
        return text_content
    except Exception as e:
        return f"Failed to process TXT file: {e}"


def process_ppt(file_bytes: bytes):
    """Process PPT file to extract text and image descriptions, integrating them into the text content."""
    text_content = ""
    image_descriptions = {}
    ppt = Presentation(BytesIO(file_bytes))
    
    for slide_num, slide in enumerate(ppt.slides):
        slide_text = ""
        for shape in slide.shapes:
            if shape.has_text_frame:
                slide_text += shape.text + "\n"
        text_content += f"Slide {slide_num + 1}:\n{slide_text}\n"
        
        for shape in slide.shapes:
            if shape.shape_type == 13:  # 13 is the shape type for images in PPTX
                image = shape.image.blob
                image_filename = f"slide_{slide_num + 1}_img_{len(image_descriptions) + 1}.jpg"
                description = get_image_description(image)
                image_descriptions[image_filename] = description
                text_content += f"\n[Image: {image_filename}]\nDescription: {description}\n"
    
    status = ingest_document(text_content)
    return text_content