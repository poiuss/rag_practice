import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PERSIST_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "rag_docs"

pdf_files = [
    DATA_DIR / "2040_report.pdf",
    DATA_DIR / "OneNYC-2050-Summary.pdf",
]

for p in pdf_files:
    if not p.exists():
        raise FileNotFoundError(f"PDF 파일이 없습니다: {p}")

documents = []
for pdf_path in pdf_files:
    documents.extend(PyPDFLoader(str(pdf_path)).load())

print("로드된 페이지 수:", len(documents))
if len(documents) == 0:
    raise RuntimeError("PDF 로딩 결과가 0페이지입니다.")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(documents)

print("생성된 청크 수:", len(splits))
if len(splits) == 0:
    raise RuntimeError("청킹 결과가 0개입니다.")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 임베딩 테스트는 여기에서 먼저
embeddings.embed_query("test")
print("임베딩 호출 OK")

# DB 폴더 완전 초기화
if PERSIST_DIR.exists():
    shutil.rmtree(PERSIST_DIR)
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

db = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=str(PERSIST_DIR),
    collection_name=COLLECTION_NAME,
)

count = db._collection.count()
print("문서 수:", count)

if count == 0:
    raise RuntimeError("Chroma에 문서가 저장되지 않았습니다. (count=0)")

# db.persist()  # 최신 조합에서는 필요 없거나 존재하지 않습니다.

print("Chroma DB 생성 완료")
print("저장 경로:", PERSIST_DIR)
print("컬렉션 이름:", COLLECTION_NAME)