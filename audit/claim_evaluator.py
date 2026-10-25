import json

from agent.llm_client import ask_llm

def evaluate_faithfulness(question,context,answer):
    """
    评估大模型回答的忠诚度（Faithfulness）
    """
    print(f"\n[裁判介入] 正在进行忠实度审查...")

    prompt=f"""
    你是一个严格的 AI 质检员。请根据提供的【参考上下文】和【用户问题】，评估【模型回答】的质量。
    
    【参考上下文】：
    {context}
    
    【用户问题】：
    {question}
    
    【模型回答】：
    {answer}
    
    请按以下标准打分并给出理由：
    1. 忠实度（0-10分）：回答是否完全基于参考上下文？如果有任何上下文里没提过的内容（幻觉），直接给 0 分。
    2. 如果回答是“抱歉，知识库中未记录”，只要上下文确实没有答案，忠实度给 10 分。

    【输出格式要求】：
    请务必只输出一个 JSON 格式的字符串，不要包含任何 Markdown 标记（如 ```json ），不要包含多余的解释。
    格式如下：
    {{"score": 分数(整数), "reason": "打分的简短理由"}}
    """

    raw_reply=ask_llm(prompt)

    try:
        clean_reply=raw_reply.replace("```json",'').replace("```","").strip()
        result=json.loads(clean_reply)
        print(f"[裁判结果] 忠实度得分: {result['score']} / 10")
        print(f"[裁判理由] {result['reason']}")
        return result
    except Exception as e:
        print(f"[裁判解析失败] 大模型返回了无法解析的内容: {raw_reply}")
        return{"score":0,"reason":"解析失败"}
