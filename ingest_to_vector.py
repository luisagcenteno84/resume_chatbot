from langchain_community.vectorstores.chroma import Chroma
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
#from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = Chroma("resume_rag", embeddings)

#loader = WebBaseLoader("https://www.linkedin.com/in/luiscgonzalez/")
loader = PyPDFLoader(file_path="resume.pdf")
data = loader.load()
#print(data)

text_splitters = RecursiveCharacterTextSplitter(chunk_size=250,chunk_overlap=50)
all_splits = text_splitters.split_documents(data)

#print(all_splits)

vector_store.add_documents(all_splits)

print(vector_store.similarity_search("Databricks"))
