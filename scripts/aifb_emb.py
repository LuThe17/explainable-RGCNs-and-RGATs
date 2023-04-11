import pandas as pd
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.walkers import RandomWalker
from pykeen.triples import TriplesFactory
from rdflib import Graph, Literal, URIRef
import rdflib
import numpy as np
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import pickle
import collections
import csv
from pykeen.pipeline import pipeline
from rdflib.namespace import RDF
from collections import Counter



homedir = "C:/Users/luisa/Projekte/Masterthesis/AIFB"
def kg_to_tsv(g):
    df = pd.DataFrame(columns=['subject','predicate','object'])
    #g = Graph()
    #g.parse(data)
    for s, p, o in g:
        if isinstance(o, Literal):
            continue
        df = pd.concat([df, pd.DataFrame([[s,p,o]], columns=['subject','predicate','object'])])
    df.to_csv(homedir +"/data/aifb_renamed_bn.tsv", sep="\t", index=False, header=None)
    return df

def create_rdf2vec_embedding(kg, entities):
    kg = KG(kg)
    transformer = RDF2VecTransformer(
        Word2Vec(epochs=250), # mehr Epochen (100/200), mehr als 75% accuracy
        walkers=[RandomWalker(1, 2, with_reverse=False, n_jobs=2)],
    )
    embeddings, literals = transformer.fit_transform(kg, entities)  # gesamter Graph
    print(embeddings)
    emb_train = embeddings[:140]
    emb_test = embeddings[140:]
    with open(homedir + "/data/train_embedding", "wb") as fp:   #Pickling
        pickle.dump(emb_train, fp)
    with open(homedir + "/data/test_embedding", "wb") as fp:   #Pickling
        pickle.dump(emb_test, fp)
    return emb_train, emb_test, embeddings

def save_rdf2vec_emb(emb_train, emb_test, emb):
    with open(homedir + "/data/train_rdf_embedding", "wb") as fp:   #Pickling
        pickle.dump(emb_train, fp)
    with open(homedir + "/data/test_rdf_embedding", "wb") as fp:   #Pickling
        pickle.dump(emb_test, fp)
    with open(homedir + "/data/rdf_embedding", "wb") as fp:   #Pickling
        pickle.dump(emb, fp)

def create_pykeen_embedding(train, test, entities):
    
    result = pipeline(
        training=train,
        testing=test,
        model="TransE",
        device="cpu",
        epochs=200,
    )
    model = result.model

    entity_embeddings = model.entity_representations[0](indices=None).cpu().detach().numpy()

    entity_to_id_dict = result.training.entity_to_id

    word_vectors = {}

    for k, v in entity_to_id_dict.items():
        word_vectors[k] = entity_embeddings[v]

    embeddings = []
    for node in entities:
        embeddings.append(word_vectors[node])#.toPython()])
    print(embeddings)
    embeddings_np = np.array(embeddings)

    pykeen_emb_train = embeddings_np[:140]
    pykeen_emb_test = embeddings_np[140:]
    return pykeen_emb_train, pykeen_emb_test, entity_embeddings

def save_pykeen_emb (emb_train, emb_test, emb):
    with open(homedir + "/data/embeddings/train_pykeen_embedding", "wb") as fp:   #Pickling
        pickle.dump(emb_train, fp)
    with open(homedir + "/data/embeddings/test_pykeen_embedding", "wb") as fp:   #Pickling
        pickle.dump(emb_test, fp)
    with open (homedir + "/data/embeddings/pykeen_embedding", "wb") as fp:
        pickle.dump(emb, fp)

def remove_aff_mem_emp(homedir, graph):
    data=pd.read_csv(homedir + '/data/trainingSet.tsv',sep='\t')
    test = pd.read_csv(homedir + '/data/testSet.tsv', sep='\t')
    g = Graph()
    g.parse(graph)

    print(len(g))

    l = []
    for row in data.index:
        person = rdflib.term.URIRef(data.person[row])
        ag = rdflib.term.URIRef(data.label_affiliation[row])
        aff = rdflib.term.URIRef('http://swrc.ontoware.org/ontology#affiliation')
        mem = rdflib.term.URIRef('http://swrc.ontoware.org/ontology#member')
        emp = rdflib.term.URIRef('http://swrc.ontoware.org/ontology#employs')
        for s, p, o in g:
            if s == person and p == aff  and o == ag:
                l.append(s)
                l.append(p)
                l.append(o)
                g.remove((person, aff, ag))
            if s == ag and p == mem  and o == person:
                l.append(s)
                l.append(p)
                l.append(o)
                g.remove((ag, mem, person))
            if s == ag and p == emp  and o == person:
                l.append(s)
                l.append(p)
                l.append(o)
                g.remove((ag, emp, person))
    for row in test.index:
        person = rdflib.term.URIRef(data.person[row])
        ag = rdflib.term.URIRef(data.label_affiliation[row])
        aff = rdflib.term.URIRef('http://swrc.ontoware.org/ontology#affiliation')
        mem = rdflib.term.URIRef('http://swrc.ontoware.org/ontology#member')
        emp = rdflib.term.URIRef('http://swrc.ontoware.org/ontology#employs')
        for s, p, o in g:
            if s == person and p == aff  and o == ag:
                l.append(s)
                l.append(p)
                l.append(o)
                g.remove((person, aff, ag))
            if s == ag and p == mem  and o == person:
                l.append(s)
                l.append(p)
                l.append(o)
                g.remove((ag, mem, person))
            if s == ag and p == emp  and o == person:
                l.append(s)
                l.append(p)
                l.append(o)
                g.remove((ag, emp, person))
    return g

def remove_literal_in_graph(g):
    l = []
    for s, p, o in g:
        if type(o) == rdflib.term.Literal:
            l.append(s)
            l.append(p)
            l.append(o)
            g.remove((s, p, o))
    return g

def rename_bnode_in_graph(g):
    new_iri = URIRef("http://bnode.org/")
    for s, p, o in g:
    
        if isinstance(o, rdflib.BNode):
            
            o_iri = URIRef(f"{new_iri}{o}")
            g.remove((s, p, o))#
            g.add((s, p, o_iri))
            #g.add((o_iri, RDF.subject, o))
            o = o_iri

        if isinstance(s, rdflib.BNode):
            s_iri = URIRef(f"{new_iri}{s}")
            g.remove((s, p, o))
            g.add((s_iri, p, o))
            #g.add((s_iri, RDF.subject, s))
            s = s_iri
        
    return g

def SVM_classifier(train_emb, test_emb, traindata, testdata):
    SVM_classifier = svm.SVC(kernel='rbf', C=1.0, random_state=42)
    print(len(train_emb))
    print(traindata["label_affiliation"].shape)
    SVM_classifier.fit(train_emb, traindata["label_affiliation"])
    predictions = SVM_classifier.predict(test_emb)
    score = SVM_classifier.score(test_emb, testdata["label_affiliation"])
    return predictions, score

def Gaussian_classifier(train_emb, test_emb, traindata, testdata):
    Gaussian_classifier = GaussianProcessClassifier(1.0 * RBF(1.0))
    Gaussian_classifier.fit(train_emb, traindata["label_affiliation"])
    predictions = Gaussian_classifier.predict(test_emb)
    score = Gaussian_classifier.score(test_emb, testdata["label_affiliation"])
    return predictions, score

def st(node):
    """
    Maps an rdflib node to a unique string. We use str(node) for URIs (so they can be matched to the classes) and
    we use .n3() for everything else, so that different nodes don't become unified.

    Source: https://github.com/pbloem/gated-rgcn/blob/1bde7f28af8028f468349b2d760c17d5c908b58b/kgmodels/data.py#L16
    """
    if type(node) == URIRef:
        return str(node)
    else:
        return node.n3()
    

if __name__ == '__main__':
    homedir = "C:/Users/luisa/Projekte/Masterthesis/AIFB"
    
    kg = homedir + '/data/aifb_renamed_bn.nt'
    g = Graph()
    g = g.parse(kg)
    # nodes = set()
    # relations = Counter()

    # for s, p, o in g:
    #     nodes.add(st(s))
    #     nodes.add(st(o))

    #     relations[st(p)] += 1

    # i2n = list(nodes)
    # kg = remove_aff_mem_emp(homedir, kg)
    # kg = remove_literal_in_graph(kg)

    # kg = rename_bnode_in_graph(kg)
    # kg.serialize(format="nt", destination= homedir +"/data/aifb_renamed_bn.nt")


    #df = kg_to_tsv(g)
   
    traindata = pd.read_csv(homedir +"/data/trainingSet.tsv", sep="\t") # train und test zusammen
    testdata = pd.read_csv(homedir + "/data/testSet.tsv", sep="\t")
    entities = traindata['person'].append(testdata['person'])
    testpy = testdata[:2]
    pykeen_data = TriplesFactory.from_path(homedir + "/data/aifb_renamed_bn.tsv", sep="\t")
    pykeen_test = TriplesFactory.from_path(homedir + "/data/testSetpy.tsv", sep="\t")
    pykeen_emb_train, pykeen_emb_test, pykeen_embeddings  = create_pykeen_embedding(pykeen_data, pykeen_test, entities)
    # train_emb, test_emb, rdf2vec_embeddings = create_rdf2vec_embedding(kg, entities)
    save_pykeen_emb(pykeen_emb_train, pykeen_emb_test, pykeen_embeddings)
    # #save_rdf2vec_emb(train_emb, test_emb, rdf2vec_embeddings)
    # #pred_rdf_G, score_rdf_G = Gaussian_classifier(train_emb, test_emb, traindata, testdata)
    # pred_rdf_py, score_py_G = Gaussian_classifier(pykeen_emb_train, pykeen_emb_test, traindata, testdata) # size:140x50
    # #pred_rdf_SVM, score_rdf_SVM = SVM_classifier(train_emb, test_emb, traindata, testdata)
    # pred_py_SVM, score_py_SVM = SVM_classifier(pykeen_emb_train, pykeen_emb_test, traindata, testdata)
    # #np.savetxt(homedir + "/data/results/prediction_Gaussianclassifier.txt", pred,fmt="%s")
    # print('Score_rdf_Gaussian Kernel: ', score_rdf_G)
    # print('Score_pykeen_Gaussian Kernel: ', score_py_G)
    # print('Score_rdf_SVM: ', score_rdf_SVM)
    # print('Score_pykeen_SVM: ', score_py_SVM)
    



    
    

