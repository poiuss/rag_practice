from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from dotenv import load_dotenv

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY가 설정되어 있어야 합니다."

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

one_nyc_path = data_dir / "OneNYC-2050-Summary.pdf"
seoul_2040_path = data_dir / "2040_report.pdf"

one_loader = PyPDFLoader(str(one_nyc_path))
seoul_loader = PyPDFLoader(str(seoul_2040_path))

one_docs = one_loader.load()
seoul_docs = seoul_loader.load()

print(len(one_docs), len(seoul_docs), type(one_docs[0]).__name__, one_docs[0].metadata)

# (6) 청킹 (오버랩 없음)
splitter_no_overlap = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)

one_splits_no = splitter_no_overlap.split_documents(one_docs)
print(len(one_splits_no), type(one_splits_no[0]).__name__)

# (7) 앞쪽 2개 청크 출력
for i in range(2):
    print(f"[chunk {i}]")
    print(one_splits_no[i].page_content[:600])
    print("metadata:", one_splits_no[i].metadata)
    print()

# (8) all_splits 생성
all_splits = one_splits_no + splitter_no_overlap.split_documents(seoul_docs)
print(type(all_splits), type(all_splits[0]), all_splits[0].metadata)

# (9) 서울 문서 청킹
seoul_splits_no = splitter_no_overlap.split_documents(seoul_docs)
print(len(seoul_splits_no))

# (10) 오버랩 없을 때 경계 확인
idx = 5
print(seoul_splits_no[idx].page_content[-350:])
print()
print(seoul_splits_no[idx+1].page_content[:350])

# (11) 오버랩 있는 Splitter
splitter_with_overlap = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

seoul_splits_ov = splitter_with_overlap.split_documents(seoul_docs)
print(len(seoul_splits_ov))

# (12) 오버랩 적용 후 경계 확인
idx = 5
print(seoul_splits_ov[idx].page_content[-350:])
print()
print(seoul_splits_ov[idx+1].page_content[:350])


# (1) 임베딩 객체 생성
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# (2) Chroma 벡터 DB 생성 및 문서 적재
from langchain_chroma import Chroma

persist_dir = "chroma_db"

vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
    persist_directory=persist_dir
)

# (3) 유사도 검색 테스트
query = "도시기본계획에서 핵심 전략은 무엇인가?"
docs = vectorstore.similarity_search(query, k=3)

for i, d in enumerate(docs):
    print(f"[{i}] page:", d.metadata.get("page"))
    print(d.page_content[:500])
    print()

# (4) 벡터 길이 확인
v = embeddings.embed_query("테스트 문장입니다.")
print(len(v))