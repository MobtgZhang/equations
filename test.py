import networkx as nx

def analyze_graph(G, p1=0.3, p2=0.7):
    # 1. 删除低权重边
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < p1]
    G.remove_edges_from(edges_to_remove)

    # 2. 孤立点
    isolated_nodes = list(nx.isolates(G))

    # 3. 连通分量
    components = [list(c) for c in nx.connected_components(G)]

    isolated_clusters = []
    duplicate_clusters = []

    # 4. 分析每个连通分量
    for comp in components:
        subgraph = G.subgraph(comp)

        # 如果只剩一个点，不处理（孤立点已经提取过）
        if len(subgraph) == 1:
            continue

        # 检查边的权重分布
        weights = [d['weight'] for _, _, d in subgraph.edges(data=True)]

        if all(p1 <= w < p2 for w in weights):  
            isolated_clusters.append(comp)   # 弱别名簇
        elif all(p2 <= w < 1.0 for w in weights):
            duplicate_clusters.append(comp)  # 强别名簇
        else:
            # 混合情况：可能有强有弱，可以再细分
            isolated_clusters.append(comp)

    return {
        "isolated_nodes": isolated_nodes,
        "isolated_clusters": isolated_clusters,
        "duplicate_clusters": duplicate_clusters
    }
