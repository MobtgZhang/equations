import os
import json
import argparse
import pickle
import pathlib
import torch
from transformers import AutoTokenizer,AutoModel
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
def get_embeddings(words_list,model,tokenizer, batch_size,device):
    embeddings = []

    for i in tqdm(range(0, len(words_list), batch_size)):
        batch_words = words_list[i:i+batch_size]

        # 编码文本
        inputs = tokenizer(batch_words, padding=True, truncation=True,
                           return_tensors="pt", max_length=16)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # 取 [CLS] 向量作为句子/标签的embedding
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        embeddings.append(cls_embeddings.cpu())

    # 拼接结果
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

        if 0.0 < sim < p1:
            relation = "isolated"
        elif p1 <= sim < p2:
            relation = "alias"
        elif p2 <= sim <= 1.0:
            relation = "duplicate"
        else:
            continue
        edges.append((word_a, word_b, {"weight": sim, "relation": relation}))
    return edges

def build_label_graph_multiple(words_list, all_words2index,embeddings, p1=0.6, p2=0.9, workers=None, chunk_size=5000):
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
def analyze_graph(G):
    isolated_edges = [(u, v) for u, v, d in G.edges(data=True) if d["relation"] == "isolated"]
    alias_edges = [(u, v) for u, v, d in G.edges(data=True) if d["relation"] == "alias"]
    duplicate_edges = [(u, v) for u, v, d in G.edges(data=True) if d["relation"] == "duplicate"]

    print("📌 别名聚合簇：")
    alias_subgraph = G.edge_subgraph(alias_edges)
    for comp in nx.connected_components(alias_subgraph):
        print(" -", comp)

    print("\n📌 需要去掉的重复标签：")
    for u, v in duplicate_edges:
        print(f" - {u} <-> {v}")

    print("\n📌 孤立标签：")
    alias_and_dup_nodes = set(sum(alias_edges + duplicate_edges, ()))
    for node in G.nodes():
        if node not in alias_and_dup_nodes:
            print(" -", node)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",type=str,default="/mnt/data/kw/models/Qwen/Qwen3-32B")
    parser.add_argument("--config_path",type=str,default="./config")
    parser.add_argument("--batch_size",type=int,default=128)
    parser.add_argument("--embedding_model",type=str,default="/mnt/data/kw/models/google-bert/bert-large-cased")
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
    bert_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
    bert_model = AutoModel.from_pretrained(args.embedding_model).to(device)
    
    # 调用函数获取 embedding
    word2embedding = get_embeddings(all_words_list,bert_model,bert_tokenizer,args.batch_size,device)
    del bert_tokenizer
    del bert_model
    print("Embedding 矩阵大小:", word2embedding.shape)
    
    graph_path = os.path.join(args.result_path,model_name,"graph")
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)
    with os.scandir(graph_path) as it:
        if not any(it):
            # 创建关系图，阈值设置为0.5，0.9
            p1 = 0.65
            p2 = 0.92
            all_graph_list = []
            for index in tqdm(all_result_dict,desc="processing"):
                words_list = all_result_dict[index]
                G = build_label_graph_multiple(words_list,all_words2index,word2embedding,p1=p1, p2=p2)
                save_graph_file = os.path.join(graph_path,f"{index}_graph.gpickle")
                # 保存
                with open(save_graph_file, 'wb') as f:
                    pickle.dump(G, f)
                all_graph_list.append(G)
        else:
            # 读取
            all_graph_list = []
            for file_name in os.listdir(graph_path):
                load_graph_file = os.path.join(graph_path,file_name)
                # Load graph
                with open(load_graph_file, 'rb') as f:
                    G = pickle.load(f)
                all_graph_list.append(G)
    # 分析图
    # analyze_graph(G)
    # 随后使用大模型对标签体系进行优化处理
    # 针对于每个簇，① 重复的标签请大模型归纳为一个标签；② 别名标签则用大模型进行名称优化
    
if __name__ == "__main__":
    main()
    # G = build_label_graph(words_list, p1=0.3, p2=0.7)
    # analyze_graph(G)
