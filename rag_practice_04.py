# 실습5: 질의 확장 구현하기 (1~6)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag_practice_03 import retriever, format_docs, answer_chain


# (1) 문자열 출력 파서
str_parser = StrOutputParser()

# (2) 질문 재작성용 LLM
rewrite_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 검색용 질문 재작성기입니다. "
     "사용자의 질문을 문서 검색에 적합하도록 더 구체적으로 바꿉니다. "
     "출력은 한 줄의 한국어 질문 1개만 반환합니다."),
    ("user", "원 질문: {question}\n\n재작성 질문:")
])

# (3) 체인 생성
rewrite_chain = rewrite_prompt | rewrite_llm | str_parser


# (4) 질문 확장
original_q = "도시기본계획 방향이 뭐야?"
expanded_q = rewrite_chain.invoke({"question": original_q})

print("원 질문:", original_q)
print("확장 질문:", expanded_q)
print()


# (5) 확장 질문으로 검색
docs = retriever.invoke(expanded_q)

for i, d in enumerate(docs):
    print(f"[chunk {i}] page={d.metadata.get('page')}")
    print(d.page_content[:450])
    print()


# (6) 확장 질문 + 컨텍스트로 답변 생성
context = format_docs(docs)
final_answer = answer_chain.invoke({"question": expanded_q, "context": context})

print("=== FINAL ANSWER ===")
print(final_answer)