import streamlit as st
from sentence_transformers import CrossEncoder

# 缓存加载 cross-encoder 模型
@st.cache_resource
def load_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """
    加载用于 reranking 的 CrossEncoder。
    """
    return CrossEncoder(model_name)

def rerank_documents(query: str, docs: list[dict], reranker) -> list[dict]:
    """
    对第一阶段检索出的 docs 进一步 rerank。
    Args:
      - query: 用户原始查询
      - docs:  List[{"title":..., "abstract":..., ...}]
      - reranker: CrossEncoder 实例
    Returns:
      - 排序后的 docs 列表（相关度最高的在前）
    """
    # 构建 (query, 文档段落) 对
    pairs = [(query, doc["abstract"]) for doc in docs]
    # 得到每个 pair 的相关度分数
    scores = reranker.predict(pairs)
    # 按分数降序排序
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    # 返回排序后的文档
    return [doc for _, doc in ranked]
