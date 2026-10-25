import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import chromadb
from chromadb.utils import embedding_functions


class Retriever:
    def __init__(self,data_path='data/company_rules.txt'):
        self.client=chromadb.Client()

        self.embedding_fn=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-zh-v1.5"
        )
        self.collection=self.client.get_or_create_collection(
            name='company_rules',
            embedding_function=self.embedding_fn,
        )

        with open(data_path,'r',encoding='utf-8') as f:
            text=f.read()

        chunks=[line.strip() for line in text.split('\n') if line.strip()]

        self.collection.upsert(
            documents=chunks,
            ids=[f'doc_{i}' for i in range(len(chunks))]
        )
        print(f"[知识库] 初始化完成，共加载了 {len(chunks)} 条规则。")

    def search(self,query,top_k=1):
        """检索与用户问题最相关的知识块"""
        print(f" [检索] 正在寻找: {query}")

        results=self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        return results['documents'][0]

if __name__=='__main__':
    retriever=Retriever()

    print("\n--- 测试 1：查询知识库里有的内容 ---")
    ans1=retriever.search('年假有几天？')
    print(f"检索结果: {ans1}")

    print("\n--- 测试 2：查询知识库里完全没有的内容 ---")
    ans2 = retriever.search("病假怎么休？")
    print(f"检索结果: {ans2}")