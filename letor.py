import os

def set_up_data():
    os.system("wget -c https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/nfcorpus.tar.gz -P data")
    os.system("tar -xvf data/nfcorpus.tar.gz")  

def main():
    set_up_data()
