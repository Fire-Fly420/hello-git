# main.py
from keyword_extraction import extract_keywords_tfidf,dynamic_stop
from semantic_analysis import batch_embed #batch_embed负责把关键词列表批量变成768维向量矩阵
from sklearn.cluster import KMeans #KMeans常用的无监督聚类算法
import numpy as np

if __name__ == '__main__':
    # ----------------------------------------------------------
    # 阶段 1：用历史笔记训练动态停用词 + 语义聚类测试
    # ----------------------------------------------------------
    # 1) 读取历史笔记，可从文件/数据库加载
    historical_notes = [
        "今天高等数学课讲了泰勒展开，重点记忆了余项公式",
        "项目周会确定了下周三评审，需要准备原型图和需求文档",
        "这篇论文提出了一种新的卷积神经网络结构用于图像分割",
        # …… 请确保 >=100 条，这里只是示例
    ]  # <-- 这里替换成真实数据

    # 2) 训练动态停用词并缓存
    dynamic_stop.fit(historical_notes)
    print("✅ 动态停用词已生成并缓存")

    # 3) 关键词提取 + 语义向量化 + KMeans 聚类
    k = 6  # 预估类别数，可多次尝试
    keywords_batch = extract_keywords_tfidf(historical_notes, top_n=4)
    X = batch_embed(keywords_batch) #对整批keywords跑一遍中文句向量模型，得到一个二维浮点矩阵--每行就是一篇笔记的“语义坐标”
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)

    # 4) 打印每个簇下的典型关键词，人工快速验证
    print("\n===== 聚类结果预览 =====")
    for cluster_id in range(k):
        # 找到属于该簇的所有笔记索引
        idx_in_cluster = np.where(labels == cluster_id)[0]
        # 打印前 5 条笔记的关键词做肉眼检查
        print(f"\n【簇 {cluster_id}】共 {len(idx_in_cluster)} 条笔记")
        for i in idx_in_cluster[:5]:
            print("   ", " ".join(keywords_batch[i]))

    # ----------------------------------------------------------
    # 阶段 2：交互式关键词提取（保持原有功能）
    # ----------------------------------------------------------
    while True:
        text = input("\n请输入新笔记内容（直接回车退出）：\n").strip()
        if not text:
            break
        kw = extract_keywords_tfidf([text], top_n=4)[0]
        print("关键词：", kw)