# -*- coding: utf-8 -*-
"""
Created on Sun May 17 17:00:27 2020

@author: 63561
"""

import os, sys
import random
import tqdm

rand_seed = 1726
dir = "../data/"
paths = ["qss_tab.txt", "qts_tab.txt"]
download_url = "https://disk.pku.edu.cn:443/link/2A162D0CB82424AF9E62113192ED75BA"

def read_raw_data():
    
    eles = set([])
    lines = []
    for _p in paths:
        p = os.path.join(dir, _p)
        assert os.path.isfile(p), \
            "Cannot find "+_p+" in "+dir+"\n.Please download the dataset from "+download_url
        with open(p, "r", encoding="utf-8") as f:
            _lines = f.readlines()[1:]
            lines += _lines
    for l in tqdm.tqdm(lines):
        ele = l.strip().split("\t")
        eles.add((ele[1], ele[2], ele[3]))
    eles = list(eles)
    title = [e[0] for e in eles]
    author = [e[1] for e in eles]
    body = [e[2] for e in eles]
    return {"title": title, "author": author, "body": body}

def split(d,  n_dev=10000, n_test=10000):
    
    idxs = random.sample(list(range(len(d["title"]))), len(d["title"]))
    train = {"idx": [], "title": [], "author": [], "body": []}
    dev = {"idx": [], "title": [], "author": [], "body": []}
    test = {"idx": [], "title": [], "author": [], "body": []}
    n_train = len(d["title"]) - n_dev - n_test
    n_dev = len(d["title"]) - n_test
    print ("\ttrain ...")
    for i in tqdm.tqdm(idxs[:n_train]):
        train["idx"].append(i)
        train["title"].append(d["title"][i])
        train["author"].append(d["author"][i])
        train["body"].append(d["body"][i])
    print ("\tdev ...")
    for i in tqdm.tqdm(idxs[n_train:n_dev]):
        dev["idx"].append(i)
        dev["title"].append(d["title"][i])
        dev["author"].append(d["author"][i])
        dev["body"].append(d["body"][i])
    print ("\ttest ...")
    for i in tqdm.tqdm(idxs[n_dev:]):
        test["idx"].append(i)
        test["title"].append(d["title"][i])
        test["author"].append(d["author"][i])
        test["body"].append(d["body"][i])
    return {"train": train, "dev": dev, "test": test}

def save(d):
    
    for s in ["train", "dev", "test"]:
        print ("\t"+s+" ...")
        with open(os.path.join(dir, s+".txt"), "w", encoding="utf8") as f:
            for t in tqdm.tqdm(d[s]["body"]):
                for w in t:
                    f.write(w+" ")
                f.write("\n")
        with open(os.path.join(dir, s+"_title.txt"), "w", encoding="utf8") as f:
            for t in tqdm.tqdm(d[s]["title"]):
                for w in t:
                    f.write(w+" ")
                f.write("\n")
        with open(os.path.join(dir, s+"_author.txt"), "w", encoding="utf8") as f:
            for t in tqdm.tqdm(d[s]["author"]):
                for w in t:
                    f.write(w+" ")
                f.write("\n")

if __name__ == "__main__":
    
    random.seed(rand_seed)
    
    print ("READ RAW DATA")
    d = read_raw_data()

    print ("SPLIT TRAIN, DEV & TEST")
    d = split(d)
    
    print ("SAVE TRAIN, DEV & TEST")
    save(d)