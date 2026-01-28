import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# (1) 사전 조건: API 키
load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY가 설정되어 있어야 합니다."

# (1) 사전 조건: embeddings, vectorstore_loaded, retriever 준비(실습3 재현)
persist_dir = "chroma_db"
collection_name = "rag_docs"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore_loaded = Chroma(
    persist_directory=persist_dir,
    collection_name=collection_name,
    embedding_function=embeddings
)

retriever = vectorstore_loaded.as_retriever(search_kwargs={"k": 4})

# (3) 검색 결과를 컨텍스트로 묶는 함수
def format_docs(docs):
    lines = []
    for d in docs:
        page = d.metadata.get("page")
        src = d.metadata.get("source")
        prefix = f"(source={src}, page={page})"
        lines.append(prefix + " " + d.page_content)
    return "\n\n".join(lines)

# (4) 문서 기반 답변 체인
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 문서 기반 질의응답 도우미입니다. "
     "반드시 제공된 컨텍스트에 근거해서만 답변합니다. "
     "컨텍스트에 없는 내용은 '문서에서 확인되지 않았습니다.'라고 답변합니다. "
     "가능하면 핵심을 5줄 이내로 정리합니다."),
    ("user", "질문: {question}\n\n컨텍스트:\n{context}\n\n답변:")
])

answer_chain = prompt | llm | parser

# (5) 질문 → 검색 → 답변
def ask_with_retrieval(question: str, k: int = 4):
    docs = retriever.invoke(question)
    context = format_docs(docs)
    answer = answer_chain.invoke({"question": question, "context": context})
    return answer, docs

# 테스트 실행
question = "2040 서울 도시기본계획에서 제시하는 주요 목표를 간단히 정리해 주세요."
answer, docs = ask_with_retrieval(question)

for i, d in enumerate(docs):
    print(f"[chunk {i}] source={d.metadata.get('source')} page={d.metadata.get('page')}")
    print(d.page_content[:500])
    print()

print("=== ANSWER ===")
print(answer)

# 2. 메시지 저장하고 답변 출력하기
chat_history = []

question = "2040 서울 도시기본계획의 핵심 목표를 요약해 주세요."
answer, docs = ask_with_retrieval(question)

chat_history.append({"role": "user", "content": question})
chat_history.append({"role": "assistant", "content": answer})

print(answer)

# 3. 대화 기록 출력하기
for msg in chat_history:
    print(f"{msg['role']}: {msg['content']}\n")