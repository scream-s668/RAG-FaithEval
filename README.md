# RAG-FaithEval：RAG 忠实度评测与幻觉归因系统

## 1. 项目简介
这是一个用于评测 RAG（检索增强生成）系统回答质量与幻觉的自动化测试框架。项目使用 Python、Chroma 向量数据库和 Pytest 编写。核心目标不是去写一个多牛的 RAG 业务，而是把 RAG 系统里最难验证的“回答有没有胡编乱造”用代码量化出来。

## 2. 解决什么问题
RAG 业务落地最大的痛点是“幻觉”。通常分为两种：
1. 检索错了：知识库里没有，检索器却强行塞了一条不相关的上下文给大模型。
2. 生成错了：大模型拿到了正确的上下文，却还是自己瞎编内容。
本项目通过引入“大模型裁判（LLM-as-a-Judge）”和确定性断言，自动抓取上述问题，防止 RAG 系统在线上胡说八道。

## 3. 核心模块
- **检索模块 (`rag_pipeline/retriever.py`)**：使用 Chroma 本地向量数据库，底层接入了国产开源中文向量模型 `BAAI/bge-small-zh-v1.5`，支持文本切片和基于语义的相似度检索。
- **生成模块 (`rag_pipeline/generator.py`)**：将用户问题与检索到的上下文拼接，调用 DeepSeek API 生成回答。
- **裁判模块 (`audit/claim_evaluator.py`)**：读取上下文和回答，强制大模型输出严格的 JSON 格式打分（0-10分）。如果模型编造了上下文之外的内容，直接给 0 分。
- **测试用例 (`tests/test_faithfulness.py`)**：使用 Pytest 驱动的自动化断言矩阵。

## 4. 如何运行
1. 安装依赖：
   `pip install pytest chromadb sentence-transformers openai`
2. 配置密钥：
   在 `agent/` 目录下新建 `secret_key.py` 文件，填入你的 DeepSeek Key：`REAL_API_KEY = "sk-你的密钥"`。
3. 运行测试：
   `pytest tests/test_faithfulness.py -s -v`

## 5. 测试用例说明
项目包含 3 个核心测试断言：
- `test_faithfulness_normal`：验证正常流程。大模型基于上下文正确回答，裁判断言得分必须大于等于 8 分。
- `test_faithfulness_hallucination_caught`：测试抓幻觉能力。人为伪造一段知识库里没有的“病假回答”，断言裁判必须给出 0 分，证明裁判系统有效。
- `test_retriever_recall`：测试检索精度。查询“入职满3年”，断言向量数据库必须精准召回包含“3年”和“10天”的文档块。

## 6. CI/CD 与降级机制
- **本地环境**：使用 `os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")` 强制走国内镜像下载中文模型，保证本地跑得快。
- **云端流水线**：考虑到 GitHub Actions 拉取百兆模型太慢且容易失败，代码中利用 `pytest.mark.skipif` 加入了环境隔离判断。在云端环境自动跳过需要下载模型的检索测试，确保 CI 流水线绿灯。
- 本地预期结果：`3 passed`。云端预期结果：`2 passed, 1 skipped`。

## 7. 目录结构
```text
RAG-FaithEval/
├── agent/          # 大模型 API 封装 (llm_client) 与密钥
├── audit/          # 裁判模块 (claim_evaluator)
├── data/           # 本地知识库 (company_rules.txt)
├── rag_pipeline/   # 检索与生成管道 (retriever, generator)
└── tests/          # Pytest 自动化测试用例
```
8. 局限与边界
- 文本切块使用了最基础的按行切分，真实业务场景需要更复杂的文档解析（如 PDF/Word 分块）。
- 裁判模型偶尔会有“过拟合”倾向，打分受 Prompt 影响较大，目前只做了简单的 0/10 分兜底测试。
- 检索器只测了 Top-1 命中率，没有测 Top-K 的综合召回率。