# keyword_extraction.py
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import preprocess_text   # 引入公共预处理
from utils import DynamicStopwords  # 新增DynamicStopwords类，用于动态生成并缓存停用词表

# 全局单例（也可在main.py中注入）
dynamic_stop = DynamicStopwords()

def extract_keywords_tfidf(texts, top_n=4):
    """
    texts: list[str]，原始文本列表
    return: list[list[str]]，每篇文本的前 top_n 关键词
    """
    # 统一预处理：清洗 + 分词 + 去停用词
    processed = [' '.join(preprocess_text(t,
                                          stopwords_path=None,  # 忽略之前版本固定停用词列表
                                          remove_digits=True))
                 for t in texts]
    #如果停用词表为空（第一次），直接训练
    if not dynamic_stop.stopwords:
        dynamic_stop.fit(texts)

    #其余逻辑保持不变
    vectorizer = TfidfVectorizer(stop_words=dynamic_stop.stopwords)
    tfidf = vectorizer.fit_transform(processed)
    terms = vectorizer.get_feature_names_out()
    scores = tfidf.toarray()

    keywords = []
    for row in scores:
        top_idx = row.argsort()[::-1][:top_n]
        keywords.append([terms[i] for i in top_idx])
    return keywords