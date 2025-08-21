import os
import json
import argparse
import pickle
import pathlib
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial


os.environ["CUDA_VISIBLE_DEVICES"]="5,6" # "4,5"
# ---------------------
# 获取 embedding 的函数 (这里只是示例，实际中替换为你的embedding模型)
# 3. 批量获取embedding
def get_embeddings(all_words_list,model,batch_size,device):
    # embeddings 列表
    embeddings = []

    for i in tqdm(range(0, len(all_words_list), batch_size)):
        batch_words = all_words_list[i:i+batch_size]

        # 使用 sentence_transformers 内置的 encode 方法
        batch_embeddings = model.encode(
            batch_words,
            batch_size=len(batch_words),
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False
        )
        embeddings.append(batch_embeddings.cpu())

    # 拼接所有结果
    return torch.cat(embeddings, dim=0)

def _process_pairs(pair_chunk, words_list,word2idx, embeddings, p1, p2):
    """处理一批 (i,j) 对，返回边列表"""
    edges = []
    for i, j in pair_chunk:
        # 计算相似度
        word_a = words_list[i]
        e1 = embeddings[word2idx[word_a]].reshape(1, -1)
        word_b = words_list[j]
        e2 = embeddings[word2idx[word_b]].reshape(1, -1)

        sim = cosine_similarity(e1, e2)[0, 0]

        if 0.0 <= sim < p1:
            relation = "isolated"
        elif p1 <= sim < p2:
            relation = "alias"
        elif p2 <= sim <= 1.0:
            relation = "duplicate"
        else:
            continue
        edges.append((word_a, word_b, {"weight": sim, "relation": relation}))
    return edges

def build_label_graph_multiple(words_list, all_words2index,embeddings, p1=0.8, p2=0.9, workers=None, chunk_size=5000):
    """
    多进程构建标签图
    """
    if workers is None:
        workers = max(1, cpu_count() - 1)

    n = len(words_list)
    # 所有 (i, j) 对
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # 分块
    chunks = [pairs[k:k + chunk_size] for k in range(0, len(pairs), chunk_size)]

    # 准备部分函数，避免 lambda
    worker_func = partial(_process_pairs, words_list=words_list,word2idx=all_words2index, embeddings=embeddings, p1=p1, p2=p2)

    all_edges = []
    with Pool(processes=workers) as pool:
        for result in pool.imap_unordered(worker_func, chunks):
            all_edges.extend(result)

    # 一次性构建图
    G = nx.Graph()
    G.add_nodes_from(words_list)
    G.add_edges_from(all_edges)
    return G

# ---------------------
# 分析图
def analyze_graph(G, p1,p2):
    # 1. 删除弱关系边 (0 < weight < p1)
    to_remove = [(u, v) for u, v, d in G.edges(data=True) if 0 < d.get("weight", 0) < p1]
    G.remove_edges_from(to_remove)
    # 2. 提取连通分量
    components = list(nx.connected_components(G))

    isolated_clusters = []

    for comp in components:
        if len(comp) == 1:  
            # 孤立点
            isolated_clusters.append(list(comp)[0])
        else:
            # 孤立簇 (别名簇)
            subgraph = G.subgraph(comp).copy()
            
            # 3. 删除重复名：根据属性去重
            name_seen = set()
            nodes_to_remove = []
            for node, data in subgraph.nodes(data=True):
                label_name = data.get("name", node)  # 标签名
                if label_name in name_seen:
                    nodes_to_remove.append(node)
                else:
                    name_seen.add(label_name)
            print(list(subgraph.nodes()))
            subgraph.remove_nodes_from(nodes_to_remove)
            print(list(subgraph.nodes()))
            print("====================================")
            isolated_clusters.append(list(subgraph.nodes()))
    print("--------------------------------------------------------------")
    return isolated_clusters


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",type=str,default="/mnt/data/kw/models/Qwen/Qwen3-32B")
    parser.add_argument("--config_path",type=str,default="./config")
    parser.add_argument("--batch_size",type=int,default=128)
    parser.add_argument("--embedding_model",type=str,default="/mnt/data/kw/models/Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--result_path",type=str,default="./result")
    parser.add_argument("--level_name",type=str,default="third")
    args = parser.parse_args()
    return args
def main():
    args = get_args()
    model_name = pathlib.Path(args.model_path).name
    tags_name_list = ["third","second","first"]
    assert args.level_name in tags_name_list
    load_cluser_file = os.path.join(args.result_path,model_name,f"clusters_{args.level_name}.txt")
    
    
    graph_path = os.path.join(args.result_path,model_name,f"graph_{args.level_name}")
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)
    p1 = 0.75
    p2 = 0.9
    with os.scandir(graph_path) as it:
        if not any(it):
            print("不存在图，现在开始创建图")
            # 创建关系图，阈值设置为0.5，0.9
            # 加载数据集
            all_result_dict = {}
            all_words_list = []
            with open(load_cluser_file,mode="r",encoding="utf-8") as rfp:
                for line in rfp:
                    name,index = line.strip().split("\t")
                    if index not in all_result_dict:
                        all_result_dict[index] = []
                    all_words_list.append(name)
                    all_result_dict[index].append(name)
            all_words2index = {}
            for index,word in enumerate(all_words_list):
                all_words2index[word] = index
            # 加载embedding 模型，这里用到的是bert large embedding模型
            device_id = 0
            
            device = torch.device(f"cuda:{device_id}")
            model = SentenceTransformer(args.embedding_model)
            
            # 调用函数获取 embedding
            word2embedding = get_embeddings(all_words_list,model,args.batch_size,device)
            
            del model
            print("Embedding 矩阵大小:", word2embedding.shape)
            all_graph_dict = {}
            for index in tqdm(all_result_dict,desc="processing"):
                words_list = all_result_dict[index]
                G = build_label_graph_multiple(words_list,all_words2index,word2embedding,p1=p1, p2=p2)
                save_graph_file = os.path.join(graph_path,f"{index}_graph.gpickle")
                # 保存
                with open(save_graph_file, 'wb') as f:
                    pickle.dump(G, f)
                all_graph_dict[index] = G
        else:
            # 读取
            print("存在图，现在开始读取图")
            all_graph_dict = {}
            for file_name in os.listdir(graph_path):
                load_graph_file = os.path.join(graph_path,file_name)
                index = file_name.split("_")[0]
                # Load graph
                with open(load_graph_file, 'rb') as f:
                    G = pickle.load(f)
                all_graph_dict[index] = G
    # 分析图
    # analyze_graph(G)
    # 随后使用大模型对标签体系进行优化处理
    for index in all_graph_dict:
        # 针对于每个簇，① 重复的标签请大模型归纳为一个标签；② 别名标签则用大模型进行名称优化
        G = all_graph_dict[index]
        # 分析图
        analyze_graph(G, p1,p2)
if __name__ == "__main__":
    main()
    # G = build_label_graph(words_list, p1=0.3, p2=0.7)
    # analyze_graph(G)
