import streamlit as st
import time
import os
from rerank_utils import load_reranker, rerank_documents

# --- 新增：初始化对话历史 ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- 新增：简单的迭代检索查询构造函数 ---
def refine_query(orig_query: str, prev_answer: str) -> str:
    return orig_query + " " + prev_answer

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = './hf_cache'

# Import functions and config from other modules
from config import (
    DATA_FILE, EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME, TOP_K,
    MAX_ARTICLES_TO_INDEX, MILVUS_LITE_DATA_PATH, COLLECTION_NAME,
    id_to_doc_map
)
from data_utils import load_data
from models import load_embedding_model, load_generation_model
from milvus_utils import (
    get_milvus_client, setup_milvus_collection,
    index_data_if_needed, search_similar_documents
)
from rag_core import generate_answer

# --- Streamlit UI 设置 ---
st.set_page_config(layout="wide")
st.title("📄 医疗 RAG 系统 (Milvus Lite)")
st.markdown(f"使用 Milvus Lite, `{EMBEDDING_MODEL_NAME}`, 和 `{GENERATION_MODEL_NAME}`。")

# --- 渲染历史对话 ---
for turn in st.session_state.history:
    st.markdown(f"**用户：** {turn['user']}")
    st.markdown(f"**系统：** {turn['bot']}")

# --- 初始化与缓存 ---
milvus_client = get_milvus_client()

if milvus_client:
    collection_is_ready = setup_milvus_collection(milvus_client)
    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    generation_model, tokenizer = load_generation_model(GENERATION_MODEL_NAME)
    models_loaded = embedding_model and generation_model and tokenizer

    if collection_is_ready and models_loaded:
        raw = load_data(DATA_FILE)
        indexing_successful = False
        if raw:
            indexing_successful = index_data_if_needed(
                milvus_client, raw, embedding_model
            )
        else:
            st.warning(f"无法从 {DATA_FILE} 加载数据。跳过索引。")

        st.divider()

        # --- RAG 交互部分 ---
        if not indexing_successful and not id_to_doc_map:
            st.error("数据索引失败或不完整，且没有文档映射。RAG 功能已禁用。")
        else:
            query = st.text_input("请提出关于已索引医疗文章的问题:", key="query_input")

            if st.button("获取答案", key="submit_button") and query:
                start_time = time.time()

                # —— 第一次检索 & Re-ranking —— 
                with st.spinner("正在搜索相关文档..."):
                    ids, dists = search_similar_documents(
                        milvus_client, query, embedding_model
                    )
                if not ids:
                    st.warning("在数据库中找不到相关文档。")
                else:
                    docs = [id_to_doc_map[_id] for _id in ids if _id in id_to_doc_map]
                    reranker = load_reranker()
                    docs = rerank_documents(query, docs, reranker)

                      # —— 新增：可视化 Re-ranking 结果 —— 
                    import pandas as pd
                    # 重新计算 scores（与 rerank_documents 内部一致）
                    pairs = [(query, doc["abstract"]) for doc in docs]
                    scores = reranker.predict(pairs)
                    df = pd.DataFrame({
                        "doc_id": ids,
                        "title": [doc["title"] for doc in docs],
                        "distance": dists,
                        "rerank_score": scores
                    }).sort_values("rerank_score", ascending=False).reset_index(drop=True)
                    # 使用 Streamlit 自带表格展示
                    st.subheader("Re-ranking 结果")
                    st.dataframe(df)  # 或者 st.table(df)

                    # —— 生成答案 —— 
                    with st.spinner("正在根据上下文生成答案..."):
                        answer = generate_answer(query, docs, generation_model, tokenizer)

                    # —— 展示结果 & 保存至历史 —— 
                    st.subheader("系统回答：")
                    st.write(answer)
                    st.session_state.history.append({"user": query, "bot": answer})

                    # —— 新增：迭代检索按钮 —— 
                    if st.button("基于上次结果再检索", key="refine_button"):
                        refined_q = refine_query(query, answer)
                        with st.spinner("基于上次结果优化检索..."):
                            ids2, dists2 = search_similar_documents(
                                milvus_client, refined_q, embedding_model
                            )
                        docs2 = [id_to_doc_map[_id] for _id in ids2 if _id in id_to_doc_map]
                        docs2 = rerank_documents(refined_q, docs2, reranker)

                        st.subheader("优化检索的上下文：")
                        for i, doc in enumerate(docs2):
                            st.markdown(f"- **文档 {i+1}:** {doc['title']}")

                        with st.spinner("生成优化后的答案..."):
                            answer2 = generate_answer(
                                refined_q, docs2, generation_model, tokenizer
                            )
                        st.subheader("优化后的回答：")
                        st.write(answer2)
                        st.session_state.history.append({"user": refined_q, "bot": answer2})

                end_time = time.time()
                st.info(f"总耗时: {end_time - start_time:.2f} 秒")

    else:
        st.error("加载模型或设置 Milvus Lite collection 失败。请检查日志和配置。")
else:
    st.error("初始化 Milvus Lite 客户端失败。请检查日志。")

# --- 页脚/信息侧边栏 ---
st.sidebar.header("系统配置")
st.sidebar.markdown(f"**向量存储:** Milvus Lite")
st.sidebar.markdown(f"**数据路径:** `{MILVUS_LITE_DATA_PATH}`")
st.sidebar.markdown(f"**Collection:** `{COLLECTION_NAME}`")
st.sidebar.markdown(f"**数据文件:** `{DATA_FILE}`")
st.sidebar.markdown(f"**嵌入模型:** `{EMBEDDING_MODEL_NAME}`")
st.sidebar.markdown(f"**生成模型:** `{GENERATION_MODEL_NAME}`")
st.sidebar.markdown(f"**最大索引数:** `{MAX_ARTICLES_TO_INDEX}`")
st.sidebar.markdown(f"**检索 Top K:** `{TOP_K}`")
