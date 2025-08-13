# semantic_analysis.py 语义转换器，负责给关键词赋予可计算的语义
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer   # pip install -U sentence-transformers

model = SentenceTransformer('shibing624/text2vec-base-chinese')   # 本地 100MB，微调BERT模型，加载一个中文句向量模型。该模型可将任意中文文本映射成768维的稠密向量

def keywords_to_vector(keywords: List[str]) -> np.ndarray:
    """把关键词列表变成 768 维向量（平均池化），越接近的语义向量夹角越小"""
    if not keywords:
        # 空关键词给一个零向量
        return np.zeros(768)
    # 直接拼接成一句话也可以；也可以逐词取平均
    sentence = " ".join(keywords)
    return model.encode(sentence, normalize_embeddings=True)#输出1*768的numpy向量，可直接做余弦相似度

def batch_embed(keywords_batch: List[List[str]]) -> np.ndarray:
    """批量 embedding -> shape (n_notes, 768)"""
    return np.vstack([keywords_to_vector(kws) for kws in keywords_batch])