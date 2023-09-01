
from rank_bm25 import BM25Okapi
import argparse

class SentRetriever(object):
    def __init__(self,):
        self.all_sents = self.load_file('./wiki1m_for_simcse.txt')

        self.tokenized_corpus = [doc.split(" ") for doc in self.all_sents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, text):
        tokenized_query = text.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        ret = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i])[::-1][:150]
        sim=[]
        for idx in ret:
            sim.append(doc_scores[idx])
        return sim, ret

    @staticmethod
    def load_file(filename):
        i_set = []
        with open(filename, 'r') as f:
            for i_n in f:
                i_set.append(i_n.strip('\n'))
        return i_set
    
