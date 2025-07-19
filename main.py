# main.py
from keyword_extraction import extract_keywords_tfidf

if __name__ == '__main__':
    texts = [input("请输入笔记内容：\n")]
    keyword = extract_keywords_tfidf(texts, top_n=4)
    print("关键词：", keyword)