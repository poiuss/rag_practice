import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 0) API 키 준비
load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY가 설정되어 있어야 합니다."

# 1) PDF 로드
data_dir = Path("data")
one_nyc_path = data_dir / "OneNYC-2050-Summary.pdf"
seoul_2040_path = data_dir / "2040_report.pdf"

one_docs = PyPDFLoader(str(one_nyc_path)).load()
seoul_docs = PyPDFLoader(str(seoul_2040_path)).load()

# 2) 청킹 -> all_splits 만들기 (필수)
splitter_no_overlap = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
one_splits_no = splitter_no_overlap.split_documents(one_docs)
all_splits = one_splits_no + splitter_no_overlap.split_documents(seoul_docs)

# 3) 크로마 DB 생성/저장
persist_dir = "chroma_db"
collection_name = "rag_docs"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
    persist_directory=persist_dir,
    collection_name=collection_name
)

# 4) 저장된 DB 다시 로드
vectorstore_loaded = Chroma(
    persist_directory=persist_dir,
    collection_name=collection_name,
    embedding_function=embeddings
)

# 5) 로드 확인
test_docs = vectorstore_loaded.similarity_search("도시기본계획의 핵심 방향", k=2)
print(len(test_docs), test_docs[0].metadata)

# 6) 리트리버
retriever = vectorstore_loaded.as_retriever(search_kwargs={"k": 4})

# 7) 유사 청크 확인
question = "2040 서울 도시기본계획에서 제시하는 주요 목표를 간단히 정리해 주세요."
docs = retriever.invoke(question)

for i, d in enumerate(docs):
    print(f"[chunk {i}] page={d.metadata.get('page')}")
    print(d.page_content[:500])
    print()

# 8) 컨텍스트 묶기
def format_docs(docs):
    lines = []
    for d in docs:
        page = d.metadata.get("page")
        lines.append(f"(page={page}) {d.page_content}")
    return "\n\n".join(lines)

context_text = format_docs(docs)
print(context_text[:800])

# 9) 최종 답변 생성
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 문서 기반 질의응답 도우미입니다. 제공된 컨텍스트에 근거하여 답변합니다. 컨텍스트에 없는 내용은 '문서에서 확인되지 않았습니다'라고 답변합니다."),
    ("user", "질문: {question}\n\n컨텍스트:\n{context}\n\n답변:")
])

chain = prompt | llm | parser

answer = chain.invoke({"question": question, "context": context_text})
print(answer)

# 10) 검색+생성 한 번에
def ask_rag(question: str) -> str:
    docs = retriever.invoke(question)
    context = format_docs(docs)
    return chain.invoke({"question": question, "context": context})

print(ask_rag("OneNYC 2050 문서에서 주요 전략을 5줄 이내로 정리해 주세요."))