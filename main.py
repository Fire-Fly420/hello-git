# main.py
from keyword_extraction import extract_keywords_tfidf,dynamic_stop

if __name__ == '__main__':
    # 1. 先一次性喂饱100+条历史笔记做训练
    historical_notes = [...]   # 可从文件/数据库读取
    dynamic_stop.fit(historical_notes)
    print("动态停用词已经生成并缓存")

    # 2. 正常关键词提取
    text = input("请输入新笔记内容：\n")
    kw = extract_keywords_tfidf([text], top_n=4)
    print("关键词：", kw)