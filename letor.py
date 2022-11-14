import subprocess
from termcolor import colored, cprint

def set_up_data():
    cprint("Downloading corpus...","blue")
    proc = subprocess.Popen("wget -c https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/nfcorpus.tar.gz -P data".split())
    proc.wait()
    cprint("Extracting corpus...","blue")
    proc = subprocess.Popen("tar -xvf data/nfcorpus.tar.gz".split())
    proc.wait()
    # out, err = proc.communicate()
    # print(out, err)
    # os.system()

def main():
    set_up_data()

main()