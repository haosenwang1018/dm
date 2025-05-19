import streamlit as st
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def build_similarity_graph(id_to_embedding_map: dict, threshold: float = 0.7) -> nx.Graph:
    """
    构建基于 cosine 相似度的无向图：
    节点：每个文档 ID；
    边：当两节点相似度 >= threshold 时连接。
    """
    G = nx.Graph()
    ids = list(id_to_embedding_map.keys())
    embs = np.stack([id_to_embedding_map[i] for i in ids])
    sims = cosine_similarity(embs)
    for i, u in enumerate(ids):
        G.add_node(u)
        for j, v in enumerate(ids[i+1:], start=i+1):
            if sims[i, j] >= threshold:
                G.add_edge(u, v, weight=float(sims[i, j]))
    return G

def retrieve_graph_neighbors(graph: nx.Graph, seed_ids: list[int], hops: int = 1) -> set[int]:
    """
    从 seed_ids 开始做 BFS，扩展 hops 跳范围内的所有节点。
    返回 seed_ids 与它们的邻居集合。
    """
    neighbors = set(seed_ids)
    for u in seed_ids:
        # single_source_shortest_path_length 的 cutoff 指 hops 跳数
        sp = nx.single_source_shortest_path_length(graph, u, cutoff=hops)
        neighbors |= set(sp.keys())
    return neighbors
