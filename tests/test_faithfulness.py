import os
import pytest
from audit.claim_evaluator import evaluate_faithfulness
from rag_pipeline.retriever import Retriever
from agent.llm_client import HAS_REAL_API_KEY

RUN_LOCAL_TESTS=os.environ.get("RUN_LOCAL_TESTS", "true") == "true"

@pytest.mark.skipif(not HAS_REAL_API_KEY,reason="CI 环境无大模型 API Key，跳过忠实度裁判测试")
def test_faithfulness_normal():
    """测试正常情况：模型完全基于上下文回答，得分必须高"""
    print("\n--- 正在测试：正常回答的忠实度 ---")
    question="年假有几天？"
    context = "1. 员工入职满 1 年，享有 5 天带薪年假。"
    answer = "根据规定，入职满1年有5天年假。"

    result=evaluate_faithfulness(question,context,answer)

    assert result["score"]>=8,f"正常回答得分异常：{result['score']}"

@pytest.mark.skipif(not HAS_REAL_API_KEY,reason="CI 环境无大模型 API Key，跳过幻觉裁判测试")
def test_faithfulness_hallucination_caught():
    """测试幻觉情况：模型自己瞎编，裁判必须抓出来打 0 分"""
    print("\n--- 正在测试：抓出幻觉回答 ---")
    question = "病假怎么休？"
    context = "【员工年假规定】1. 员工入职满 1 年，享有 5 天带薪年假。"

    fake_hallucination_answer="根据公司规定，员工每月享有 3 天带薪病假。"

    result=evaluate_faithfulness(question,context,fake_hallucination_answer)

    assert result["score"]<=2,f"严重漏测！模型瞎编居然给了 {result['score']} 分！"
    print("测试通过：裁判成功抓住了大模型的幻觉！")

@pytest.mark.skipif(not RUN_LOCAL_TESTS,reason="CI 环境不加载本地向量模型，跳过检索测试")
def test_retriever_recall():
    """测试检索器：输入特定关键词，必须能精准召回对应的文档块"""
    print("\n---正在测试：检索召回准确度 ---")
    retriever=Retriever()

    question="入职满3年有几天年假？"
    results=retriever.search(question)

    top_result=results[0]
    assert "3 年" in top_result or "10 天" in top_result, f"召回失败！实际召回的是: {top_result}"
    print(f"检索召回成功，命中了: {top_result}")