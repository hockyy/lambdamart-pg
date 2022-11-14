import subprocess
from termcolor import colored, cprint
from tqdm import tqdm
import random

def set_up_data():
    cprint("Step 1: Setting up data","yellow")
    cprint("Downloading corpus...","blue")
    proc = subprocess.Popen("wget -c https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/nfcorpus.tar.gz -P data".split())
    proc.wait()
    cprint("Extracting corpus...","blue")
    proc = subprocess.Popen("tar -xvf data/nfcorpus.tar.gz".split())
    proc.wait()

class CorpusDocuments(object):

    def __init__(self, path:str):
        cprint(f"Generating corpus docs for {path}", "blue")
        self.documents = dict()
        with open(path) as file:
            for line in file:
                doc_id, content = line.split("\t")
                self.documents[doc_id] = content.split()
        cprint(f"Generated corpus docs for {path}", "green")

    def __str__(self) -> str:
        """Returns representation of corpus documents

        Returns:
            str: First doc in dictionary
        """
        return f'''\
{colored("First document in dictionary: ", "yellow")}
{str(list(self.documents.items())[0])}
'''

class CorpusQueries(object):
    
    def __init__(self, path:str):
        cprint(f"Generating corpus queries for {path}", "blue")
        self.queries = dict()
        with open(path) as file:
            for line in file:
                q_id, content = line.split("\t")
                self.queries[q_id] = content.split()
        cprint(f"Generated corpus queries for {path}", "green")

    def __str__(self) -> str:
        """Returns representation of corpus queries

        Returns:
            str: First doc in dictionary
        """
        return f'''\
{colored("First query in dictionary: ", "yellow")}
{str(list(self.queries.items())[0])}
'''

class Model(CorpusDocuments, CorpusQueries):

    NUM_NEGATIVES = 1

    def __init__(self, doc_path: str, query_path:str, qrel_path:str):
        cprint(f"Generating model", "yellow")
        CorpusDocuments.__init__(self, doc_path)
        CorpusQueries.__init__(self, query_path)
        self.q_docs_rel = dict()
        self.init_qrels(qrel_path)
        self.generate_dataset()
        cprint(f"Generated model", "yellow")
        pass
    
    def init_qrels(self, qrel_path):
        with open(qrel_path) as file:
            for line in file:
                q_id, _, doc_id, rel = line.split("\t")
                if (q_id in self.queries) and (doc_id in self.documents):
                    if q_id not in self.q_docs_rel:
                        self.q_docs_rel[q_id] = []
                    self.q_docs_rel[q_id].append((doc_id, int(rel)))


    def generate_dataset(self):
        # group_qid_count untuk model LGBMRanker
        self.group_qid_count = []
        self.dataset = []

        for q_id in self.q_docs_rel:
            docs_rels = self.q_docs_rel[q_id]
            self.group_qid_count.append(len(docs_rels) + self.NUM_NEGATIVES)
            for doc_id, rel in docs_rels:
                self.dataset.append((self.queries[q_id], self.documents[doc_id], rel))
            # tambahkan satu negative (random sampling saja dari documents)
            self.dataset.append((self.queries[q_id], random.choice(list(self.documents.values())), 0))

        # test
        print("number of Q-D pairs:", len(self.dataset))
        print("self.group_qid_count:", self.group_qid_count)
        assert sum(self.group_qid_count) == len(self.dataset), "ada yang salah"
        print(self.dataset[:2])

    def __str__(self) -> str:
        return f'''\
{CorpusDocuments.__str__(self)}
{CorpusQueries.__str__(self)}
'''

def main():
    # set_up_data()
    model = Model("nfcorpus/train.docs", "nfcorpus/train.vid-desc.queries", "nfcorpus/train.3-2-1.qrel")
    # print(training_documents)
    # print(queries_documents)
    # print(model)


main()