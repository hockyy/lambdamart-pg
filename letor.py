import subprocess
from termcolor import colored, cprint
from tqdm import tqdm

def set_up_data():
    cprint("Step 1: Setting up data","yellow")
    cprint("Downloading corpus...","blue")
    proc = subprocess.Popen("wget -c https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/nfcorpus.tar.gz -P data".split())
    proc.wait()
    cprint("Extracting corpus...","blue")
    proc = subprocess.Popen("tar -xvf data/nfcorpus.tar.gz".split())
    proc.wait()

class CorpusDocuments(object):

    def __init__(self, url:str):
        cprint(f"Generating corpus docs for {url}", "blue")
        self.documents = dict()
        with open(url) as file:
            for line in file:
                doc_id, content = line.split("\t")
                self.documents[doc_id] = content.split()
        cprint(f"Generated corpus docs for {url}", "green")

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
    
    def __init__(self, url:str):
        cprint(f"Generating corpus queries for {url}", "blue")
        self.queries = dict()
        with open(url) as file:
            for line in file:
                q_id, content = line.split("\t")
                self.queries[q_id] = content.split()
        cprint(f"Generated corpus queries for {url}", "green")

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
    def __init__(self, doc_url: str, query_url:str):
        super(CorpusDocuments, self).__init__(doc_url)
        super(CorpusQueries, self).__init__(query_url)
        print(len(self.documents.keys()))
        pass

def main():
    # set_up_data()
    # training_documents = CorpusDocuments()
    # queries_documents = CorpusQueries()
    model = Model("nfcorpus/train.docs", "nfcorpus/train.vid-desc.queries")
    # print(training_documents)
    # print(queries_documents)


main()