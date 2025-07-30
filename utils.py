import jieba
import re
import os, json, pickle
from sklearn.feature_extraction.text import TfidfVectorizer

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
        text = re.sub(r'[^\w\s]', '', text)
    
    # 去除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_stopwords(filepath='./data/stop_words.txt'):
    """
    加载停用词表。
    
    参数:
    filepath (str): 停用词表文件路径。
    
    返回:
    set: 停用词集合。
    """
    if filepath is None:
        filepath = './data/stop_words.txt' 
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords = {line.strip() for line in file if line.strip()}
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

def preprocess_text(text, stopwords_path=None, remove_digits=False):
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

class DynamicStopwords:
    """
    用 TF-IDF 的 IDF 值动态生成停用词表，并支持本地缓存
    """
    def __init__(self,
                 cache_path='./data/dynamic_stopwords.pkl',
                 idf_threshold=1.5):
        self.cache_path = cache_path
        self.idf_threshold = idf_threshold
        self.stopwords = set()

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.stopwords = pickle.load(f)
    
    def fit(self, texts):
        """
        texts: list[str]  一批原始笔记
        根据 IDF 阈值重新计算并缓存停用词
        """
        if not texts:          # 空列表直接返回
            return
        vec = TfidfVectorizer()
        vec.fit(texts)
        idfs = vec.idf_
        words = vec.get_feature_names_out()
        self.stopwords = {w for w, idf in zip(words, idfs)
                          if idf < self.idf_threshold}
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.stopwords, f)

    def is_stop(self, word):
        return word in self.stopwords