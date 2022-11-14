import subprocess
from termcolor import colored, cprint
from tqdm import tqdm
import random
import lightgbm as lgb
import numpy as np
import datetime;
import pickle

from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary

from scipy.spatial.distance import cosine
import lightgbm

def set_up_data():
    cprint("Setting up data","yellow")
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
    NUM_LATENT_TOPICS = 200

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
        cprint(f"Generating dataset", "blue")
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
        # print("number of Q-D pairs:", len(self.dataset))
        # print("self.group_qid_count:", self.group_qid_count)
        self.dictionary = Dictionary()
        self.bow_corpus = [self.dictionary.doc2bow(doc, allow_update=True) for doc in self.documents.values()]

        assert sum(self.group_qid_count) == len(self.dataset), "ada yang salah"
        cprint(f"Generated Dataset", "green")
        # print(self.dataset[:2])

    def generate_lsa(self):

        cprint(f"Generating LSA", "blue")
        self.lsi_model = LsiModel(self.bow_corpus, num_topics = self.NUM_LATENT_TOPICS) # 200 latent topics
        cprint(f"Generated LSA", "green")
        self.save_lsa()
        # test melihat representasi vector dari sebuah dokumen & query

    def save_lsa(self):
        current_filename = f'trained_model/lsa-{int(datetime.datetime.now().timestamp())}.pkl'
        cprint(f"Saving lsa to {current_filename}", "blue")
        with open(current_filename, 'wb') as f:
            pickle.dump([self.lsi_model], f)
        cprint(f"Saved", "green")

    def load_lsa(self, timestamp):
        current_filename = f'trained_model/lsa-{timestamp}.pkl'
        cprint(f"Loading lsa {current_filename}", "blue")
        with open(current_filename, 'rb') as f:
            [self.lsi_model] = pickle.load(f)
        cprint(f"Loaded", "green")

    def vector_rep(self, text):
        assert self.lsi_model != None, "Run generate or lsa model first"
        rep = [topic_value for (_, topic_value) in self.lsi_model[self.dictionary.doc2bow(text)]]
        return rep if len(rep) == self.NUM_LATENT_TOPICS else [0.] * self.NUM_LATENT_TOPICS

    def get_features(self, query, doc):
        v_q = self.vector_rep(query)
        v_d = self.vector_rep(doc)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)
        return v_q + v_d + [jaccard] + [cosine_dist]

    def get_data_set(self):

        X = []
        Y = []
        for (query, doc, rel) in self.dataset:
            X.append(self.get_features(query, doc))
            Y.append(rel)

        # ubah X dan Y ke format numpy array
        X = np.array(X)
        Y = np.array(Y)

        print(X.shape)
        print(Y.shape)
        return X, Y

    def train_model(self):

        cprint(f"Training model", "blue")
        X, Y = self.get_data_set()
        self.ranker = lightgbm.LGBMRanker(
                            objective="lambdarank",
                            boosting_type = "gbdt",
                            n_estimators = 100,
                            importance_type = "gain",
                            metric = "ndcg",
                            num_leaves = 40,
                            learning_rate = 0.02,
                            max_depth = -1)

        # di contoh kali ini, kita tidak menggunakan validation set
        # jika ada yang ingin menggunakan validation set, silakan saja
        self.ranker.fit(X, Y,
                group = self.group_qid_count,
                verbose = 10)
        cprint(f"Trained model", "green")
        self.save_model()
        # print(self.ranker.predict(X))

    def save_model(self):
        current_filename = f'trained_model/model-{int(datetime.datetime.now().timestamp())}.pkl'
        cprint(f"Saving model to {current_filename}", "blue")
        with open(current_filename, 'wb') as f:
            pickle.dump([self.ranker], f)
        cprint(f"Saved", "green")

    def load_model(self, timestamp):
        current_filename = f'trained_model/model-{timestamp}.pkl'
        cprint(f"Loading model {current_filename}", "blue")
        with open(current_filename, 'rb') as f:
            [self.ranker] = pickle.load(f)
        cprint(f"Loaded", "green")

    def predict(self, query, docs):
        assert self.lsi_model != None, "Run generate_lsa or load_lsa first"
        assert self.ranker != None, "Run train_model or load_model first"
        X_unseen = []
        for doc_id, doc in tqdm(docs):
            X_unseen.append(self.get_features(query.split(), doc.split()))

        X_unseen = np.array(X_unseen)
        scores = self.ranker.predict(X_unseen)
        return scores

    def interpret_scores(self, query, docs, scores):
        did_scores = [x for x in zip([did for (did, _) in docs], scores)]
        sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)

        print("query        :", query)
        print("SERP/Ranking :")
        for (did, score) in sorted_did_scores:
            print(did, score)

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
    # print(model.vector_rep(model.documents["MED-329"]))
    # print(model.vector_rep(model.queries["PLAIN-2435"]))
    # model.generate_lsa()
    model.load_lsa(1668446602)
    # model.train_model()
    model.load_model(1668446623)
    query = "how much cancer risk can be avoided through lifestyle change ?"

    docs =[("D1", "dietary restriction reduces insulin-like growth factor levels modulates apoptosis cell proliferation tumor progression num defici pubmed ncbi abstract diet contributes one-third cancer deaths western world factors diet influence cancer elucidated reduction caloric intake dramatically slows cancer progression rodents major contribution dietary effects cancer insulin-like growth factor igf-i lowered dietary restriction dr humans rats igf-i modulates cell proliferation apoptosis tumorigenesis mechanisms protective effects dr depend reduction multifaceted growth factor test hypothesis igf-i restored dr ascertain lowering igf-i central slowing bladder cancer progression dr heterozygous num deficient mice received bladder carcinogen p-cresidine induce preneoplasia confirmation bladder urothelial preneoplasia mice divided groups ad libitum num dr num dr igf-i igf-i/dr serum igf-i lowered num dr completely restored igf-i/dr-treated mice recombinant igf-i administered osmotic minipumps tumor progression decreased dr restoration igf-i serum levels dr-treated mice increased stage cancers igf-i modulated tumor progression independent body weight rates apoptosis preneoplastic lesions num times higher dr-treated mice compared igf/dr ad libitum-treated mice administration igf-i dr-treated mice stimulated cell proliferation num fold hyperplastic foci conclusion dr lowered igf-i levels favoring apoptosis cell proliferation ultimately slowing tumor progression mechanistic study demonstrating igf-i supplementation abrogates protective effect dr neoplastic progression"), 
       ("D2", "study hard as your blood boils"), 
       ("D3", "processed meats risk childhood leukemia california usa pubmed ncbi abstract relation intake food items thought precursors inhibitors n-nitroso compounds noc risk leukemia investigated case-control study children birth age num years los angeles county california united states cases ascertained population-based tumor registry num num controls drawn friends random-digit dialing interviews obtained num cases num controls food items principal interest breakfast meats bacon sausage ham luncheon meats salami pastrami lunch meat corned beef bologna hot dogs oranges orange juice grapefruit grapefruit juice asked intake apples apple juice regular charcoal broiled meats milk coffee coke cola drinks usual consumption frequencies determined parents child risks adjusted risk factors persistent significant associations children's intake hot dogs odds ratio num num percent confidence interval ci num num num hot dogs month trend num fathers intake hot dogs num ci num num highest intake category trend num evidence fruit intake provided protection results compatible experimental animal literature hypothesis human noc intake leukemia risk potential biases data study hypothesis focused comprehensive epidemiologic studies warranted"), 
       ("D4", "long-term effects calorie protein restriction serum igf num igfbp num concentration humans summary reduced function mutations insulin/igf-i signaling pathway increase maximal lifespan health span species calorie restriction cr decreases serum igf num concentration num protects cancer slows aging rodents long-term effects cr adequate nutrition circulating igf num levels humans unknown report data long-term cr studies num num years showing severe cr malnutrition change igf num igf num igfbp num ratio levels humans contrast total free igf num concentrations significantly lower moderately protein-restricted individuals reducing protein intake average num kg num body weight day num kg num body weight day num weeks volunteers practicing cr resulted reduction serum igf num num ng ml num num ng ml num findings demonstrate unlike rodents long-term severe cr reduce serum igf num concentration igf num igfbp num ratio humans addition data provide evidence protein intake key determinant circulating igf num levels humans suggest reduced protein intake important component anticancer anti-aging dietary interventions"), 
       ("D5", "cancer preventable disease requires major lifestyle abstract year num million americans num million people worldwide expected diagnosed cancer disease commonly believed preventable num num cancer cases attributed genetic defects remaining num num roots environment lifestyle lifestyle factors include cigarette smoking diet fried foods red meat alcohol sun exposure environmental pollutants infections stress obesity physical inactivity evidence cancer-related deaths num num due tobacco num num linked diet num num due infections remaining percentage due factors radiation stress physical activity environmental pollutants cancer prevention requires smoking cessation increased ingestion fruits vegetables moderate alcohol caloric restriction exercise avoidance direct exposure sunlight minimal meat consumption grains vaccinations regular check-ups review present evidence inflammation link agents/factors cancer agents prevent addition provide evidence cancer preventable disease requires major lifestyle")]
    scores = model.predict(query, docs)
    model.interpret_scores(query, docs, scores)
    print()


main()