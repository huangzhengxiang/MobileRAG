import pandas as pd
import json
import numpy as np
import cv2
from typing import List
import random
from tqdm import tqdm
from itertools import chain

chunk_size = 2048

def get_question(row) -> str:
    return row["question"]

def get_answer(row) -> List[str]:
    return row["answer"]["aliases"].tolist()+row["answer"]["normalized_aliases"].tolist()

def get_wiki_context(row):
    return row["entity_pages"]["wiki_context"]

def get_search_context(row):
    return row["search_results"]["search_context"]

def process(path: str = "validation-00000-of-00004.parquet",
            num: int=-1):
    questions = []
    answers = []
    docs = []
    df = pd.read_parquet(path)
    print(f"path: {path}\nrow count: {len(df)}")
    if num < 0:
        num = len(df)
    for i in tqdm(range(0, num)):
        questions.append(get_question(df.loc[i]))
        answers.append(get_answer(df.loc[i]))
        doc = ""
        for s in chain(get_wiki_context(df.loc[i]), get_search_context(df.loc[i])):
            if len(doc+s) <= chunk_size:
                doc += s
            else:
                if len(doc) > 0:
                    docs.append(doc)
                doc = s
        if len(doc) > 0:
            docs.append(doc)
    return docs, questions, answers

if __name__=="__main__":
    docs, _, _ = process()
    print(docs)