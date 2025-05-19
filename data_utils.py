import json
import streamlit as st
import re
import hashlib
import streamlit as st

def load_data(filepath):
    """Loads data from the JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        st.write(f"Loaded {len(data)} articles from {filepath}")
        return data
    except FileNotFoundError:
        st.error(f"Data file not found: {filepath}")
        return []
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from file: {filepath}")
        return []
    except Exception as e:
        st.error(f"An error occurred loading data: {e}")
        return [] 
    
def filter_documents(raw_data, min_length: int = 200):
    """
    过滤文档列表：
      1. 内容长度过滤：abstract 或 content 字段长度 < min_length 时丢弃
      2. 去重：基于 title+abstract 的 MD5 哈希去重
      3. 噪声清洗：去掉常见 HTML 残留标签和广告标记
    """
    before = len(raw_data)
    seen_hashes = set()
    cleaned = []

    for doc in raw_data:
        text = doc.get("abstract") or doc.get("content") or ""
        # 清理 HTML 残留
        text = re.sub(r'<[^>]+>', '', text)               # 去掉所有 <...> 标签
        text = re.sub(r'阅读原文|广告|点击了解更多', '', text)
        doc["abstract"] = text.strip()

        # 长度过滤
        if len(text) < min_length:
            continue

        # 去重：基于 title+text 哈希
        title = doc.get("title", "").strip()
        h = hashlib.md5((title + text).encode("utf-8")).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        cleaned.append(doc)

    after = len(cleaned)
    st.write(f"🧹 filter_documents: 原始 {before} 条 → 过滤后 {after} 条")
    return cleaned