from agent.llm_client import  ask_llm
from rag_pipeline.retriever import Retriever
from audit.claim_evaluator import evaluate_faithfulness

def ask_rag(question):
    retriever=Retriever()

    retrievied_docs=retriever.search(question)
    context="\n".join(retrievied_docs)

    print(f"\n[召回上下文]: \n{context}")

    prompt=f"""
    你是一个企业知识库助手。请根据下面提供的【参考信息】来回答用户的问题。
    如果你在参考信息中找不到答案，请直接回答“抱歉，知识库中未记录相关信息”，绝不要自己编造。
    
    【参考信息】：
    {context}
    
    【用户问题】：
    {question}
    """

    print(f"\n[大模型生成中...]")
    answer=ask_llm(prompt)

    print(f"\n[最终回答]: \n{answer}")
    return answer

if __name__ == "__main__":
    print("--- 测试 1：正常问题 ---")
    ans1=ask_rag("年假有几天？")
    context1="1. 员工入职满 1 年，享有 5 天带薪年假。"
    evaluate_faithfulness("年假有几天？",context1,ans1)

    print("\n\n--- 测试 2：知识库没有的幻觉问题 ---")
    ans2=ask_rag("病假怎么休？")
    context2="【员工年假规定】"
    evaluate_faithfulness("病假怎么休？",context2,ans2)