import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def clean_text(text, remove_digits=False):
    """
    清洗文本，去除特殊字符和多余的空格。
    
    参数:
    text (str): 输入的文本。
    remove_digits (bool): 是否去除数字，默认为False。
    
    返回:
    str: 清洗后的文本。
    """
    # 去除特殊字符，保留字母、数字和空格
    if remove_digits:
        text = re.sub(r'[^\w\s]', '', text)
    else:
        text = re.sub(r'[^\w\d\s]', '', text)
    
    # 去除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_stopwords(filepath):
    """
    加载停用词表。
    
    参数:
    filepath (str): 停用词表文件路径。
    
    返回:
    set: 停用词集合。
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords = set([line.strip() for line in file.readlines()])
    return stopwords

def remove_stopwords(tokens, stopwords):
    """
    去除停用词。
    
    参数:
    tokens (list): 分词后的单词列表。
    stopwords (set): 停用词集合。
    
    返回:
    list: 去除停用词后的单词列表。
    """
    filtered_tokens = [word for word in tokens if word not in stopwords and len(word) > 1] # 去除单字虚词，如“了”，“啊”等
    return filtered_tokens

def tokenize_text(text):
    """
    对中文文本进行分词。
    
    参数:
    text (str): 输入的文本。
    
    返回:
    list: 分词后的单词列表。
    """
    # 中文分词
    tokens = list(jieba.cut(text))
    return tokens

def preprocess_text(text, stopwords_path, remove_digits=False):
    """
    完整的中文文本预处理流程。
    
    参数:
    text (str): 输入的文本。
    stopwords_path (str): 停用词表文件路径。
    remove_digits (bool): 是否去除数字，默认为 False。
    
    返回:
    list: 预处理后的单词列表。
    """
    # 清洗文本
    cleaned_text = clean_text(text, remove_digits)
    # 加载停用词
    stopwords = load_stopwords(stopwords_path)
    # 分词
    tokens = tokenize_text(cleaned_text)
    # 去除停用词
    filtered_tokens = remove_stopwords(tokens, stopwords)
    return filtered_tokens

def extract_keywords_tfidf(texts, top_n=4):
    """
    使用TF-IDF算法提取关键词。
    
    参数:
    texts (list): 文本列表。
    top_n (int): 要提取的关键词数量，默认为10。
    
    返回:
    list: 提取的关键词列表。
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # 计算每个词的TF-IDF值
    tfidf_scores = tfidf_matrix.toarray()
    
    # 提取每个文档的关键词
    keywords = []
    for i, row in enumerate(tfidf_scores):
        sorted_indices = row.argsort()[::-1]
        doc_keywords = [feature_names[index] for index in sorted_indices[:top_n]]
        keywords.append(doc_keywords)
    
    return keywords

# 示例用法
if __name__ == "__main__":
    text = input("请输入你想要进行关键词整理的文本:\n")
    stopwords_path = './stop_words.txt'  # 使用相对路径
    processed_tokens = preprocess_text(text, stopwords_path, remove_digits=True)
    processed_text = ' '.join(processed_tokens)
    
    texts = [processed_text]  # 将预处理后的文本放入列表中
    keywords = extract_keywords_tfidf(texts, top_n=4)
    print("这段文本的关键词有:", keywords)