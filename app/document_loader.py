# app/document_loader.py
import os
import tempfile
from langchain.docstore.document import Document

def load_document(file_obj, filename):
    """
    Loads a document from an uploaded file.
    Supports PDF, TXT, and DOCX formats.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        # For PDF, use LangChain's PyPDFLoader
        from langchain.document_loaders import PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_obj.read())
            tmp.flush()
            loader = PyPDFLoader(tmp.name)
            docs = loader.load()
        os.remove(tmp.name)
        return docs

    elif ext == ".txt":
        # For text files, read directly
        content = file_obj.read().decode("utf-8")
        return [Document(page_content=content)]

    elif ext == ".docx":
        # For DOCX files, use docx2txt (make sure it's in requirements.txt)
        import docx2txt
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file_obj.read())
            tmp.flush()
            text = docx2txt.process(tmp.name)
        os.remove(tmp.name)
        return [Document(page_content=text)]

    else:
        raise ValueError("Unsupported file format")
