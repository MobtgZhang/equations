明白了 😅，6 万条原子标签量非常大，噪声标签的处理就非常关键，否则后续聚类和归纳效果都会受影响。我们可以从 **Embedding + 聚类 + 筛选** 的角度来做噪声过滤。下面我给你整理一个完整策略：

---

## 1️⃣ 噪声标签定义

* **孤立点**：语义上和其他标签差异大，embedding 距离簇中心远
* **重复/无意义标签**：像 “其他”、“测试” 之类，没有实际信息
* **异常标签**：拼写错误、乱码或混入非目标语言的标签

---

## 2️⃣ 基于 HDBSCAN 的噪声过滤

HDBSCAN 的 **天然优势**：会把不属于任何簇的点标记为 `-1`（噪声）。

### 示例：

```python
import hdbscan
from sentence_transformers import SentenceTransformer

labels = [...]  # 6万条原子标签
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(labels, show_progress_bar=True)

# HDBSCAN 聚类
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=20,   # 控制大簇最小大小
    min_samples=5,
    cluster_selection_method='eom'
)
cluster_labels = clusterer.fit_predict(embeddings)

# 筛选非噪声标签
clean_labels = [labels[i] for i in range(len(labels)) if cluster_labels[i] != -1]
noise_labels = [labels[i] for i in range(len(labels)) if cluster_labels[i] == -1]

print(f"原子标签总数: {len(labels)}")
print(f"非噪声标签: {len(clean_labels)}")
print(f"噪声标签: {len(noise_labels)}")
```

**💡 说明**：

* `min_cluster_size` 越大 → 大簇越少，孤立点越多
* `min_samples` 越大 → 聚类越严格，噪声点越多

---

## 3️⃣ 可选的增强策略

### (1) 基于 Embedding 距离阈值过滤

* 计算标签 embedding 与簇中心的余弦相似度
* 距离太远的标签认为是噪声，可单独剔除

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))

# 计算簇中心
cluster_ids = set(cluster_labels) - {-1}
cluster_centers = {}
for cid in cluster_ids:
    idxs = [i for i, x in enumerate(cluster_labels) if x == cid]
    cluster_centers[cid] = np.mean(embeddings[idxs], axis=0)

# 过滤距离簇中心太远的标签
threshold = 0.7  # 余弦相似度阈值
filtered_labels = []
for i, cid in enumerate(cluster_labels):
    if cid == -1:
        continue
    sim = cosine_similarity(embeddings[i], cluster_centers[cid])
    if sim >= threshold:
        filtered_labels.append(labels[i])
```

### (2) 基于频率或文本规则过滤

* 去掉过短或无意义标签（如长度 <2、常见占位符）
* 去掉重复或拼写错误的标签

---

## 4️⃣ 推荐流程（针对 6 万标签）

1. **生成 Transformer embedding**
2. **HDBSCAN 聚类** → 得到簇 + 天然噪声标签
3. **簇内距离过滤** → 去掉离簇中心过远的标签
4. **文本规则过滤** → 去掉长度太短或无意义标签
5. **最终输出干净原子标签** → 用于后续 L2/L1 聚类

---

✅ 这样处理后，你的原子标签数据会更加干净，噪声标签不会影响后续层级聚类和 LLM 聚类归纳。

---

如果你愿意，我可以帮你写一个 **完整 6 万原子标签噪声过滤 pipeline**，包括 **HDBSCAN 聚类 + 簇内相似度过滤 + 文本规则过滤**，直接输出干净标签集合。

你希望我帮你写吗？
