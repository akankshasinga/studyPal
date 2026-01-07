import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()
CLASS_SUBJECT_NAME = os.getenv('CLASS_SUBJECT_NAME')
DEVICE = os.getenv('DEVICE', 'cpu')  # Default to 'cpu' if not set

working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)
data_dir = f"{parent_dir}/data"
vector_db_dir = f"{parent_dir}/vector_db"
chapters_vector_db_dir = f"{parent_dir}/chapters_vector_db"

# Use a concrete HuggingFace sentence-transformers model and the configured device
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": DEVICE},
)

text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)


def vectorize_book_and_store_to_db(class_subject_name, vector_db_name):
    book_dir = f"{data_dir}/{class_subject_name}"
    vector_db_path = f"{vector_db_dir}/{vector_db_name}"
    # Recursively find PDFs in the book directory (including subfolders)
    loader = DirectoryLoader(path=book_dir, glob="**/*.pdf", loader_cls=UnstructuredFileLoader)
    documents = loader.load()
    text_chunks = text_splitter.split_documents(documents)
    Chroma.from_documents(documents=text_chunks, embedding=embedding, persist_directory=vector_db_path)
    print(f"{class_subject_name} saved to vector db: {vector_db_name}")
    return 0


def vectorize_chapters(class_subject_name):
    book_dir = f"{data_dir}/{class_subject_name}"
    # Walk the directory tree and vectorize each PDF found as a separate chapter
    for root, _, files in os.walk(book_dir):
        for file in files:
            if not file.lower().endswith('.pdf'):
                continue
            chapter_name = os.path.splitext(file)[0]
            chapter_pdf_path = os.path.join(root, file)
            loader = UnstructuredFileLoader(chapter_pdf_path)
            documents = loader.load()
            texts = text_splitter.split_documents(documents)
            persist_dir = os.path.join(chapters_vector_db_dir, chapter_name)
            Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_dir)
            print(f"{chapter_name} chapter vectorized")
    return 0