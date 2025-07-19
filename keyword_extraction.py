# keyword_extraction.py
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import preprocess_text   # 引入公共预处理

def extract_keywords_tfidf(texts, top_n=4):
    """
    texts: list[str]，原始文本列表
    return: list[list[str]]，每篇文本的前 top_n 关键词
    """
    # 统一预处理：清洗 + 分词 + 去停用词
    processed = [' '.join(preprocess_text(t)) for t in texts]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(processed)
    terms = vectorizer.get_feature_names_out()
    scores = tfidf.toarray()

    keywords = []
    for row in scores:
        top_idx = row.argsort()[::-1][:top_n]
        keywords.append([terms[i] for i in top_idx])
    return keywords