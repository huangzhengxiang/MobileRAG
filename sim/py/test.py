import os
from rag.rag import RAG
from data.trivia_qa.process import process as trivia_process

def isCorrect(ans, ans_list) -> bool:
    for a in ans_list:
        if a in ans:
            return True
    return False
    

if __name__=="__main__":
    topk=2
    test_cases=100
    path=os.path.join(".", "data", "trivia_qa", "validation-00000-of-00004.parquet")
    docs, q, a = trivia_process(path, test_cases)
    
    # rag = RAG()
    # rag.build_db(docs=docs)
    # rag.save_db()
    # from disk
    rag = RAG(from_disk=True)
    rag.buid_query_engine(topk)
    prompt="请回答：{question}，直接输出答案。"
    correct_num = 0
    for i in range(test_cases):
        ans = rag.query(prompt.format(question=q[i]))
        print(ans, a[i])
        if isCorrect(ans, a[i]):
            correct_num += 1
    print(f"Test cases: {test_cases}, Accuracy: {correct_num/test_cases}")
