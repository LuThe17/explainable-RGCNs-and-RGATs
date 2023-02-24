import pandas as pd
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker
from pykeen.triples import TriplesFactory
from rdflib import Graph, Literal
from sklearn import svm
import pickle
import rdflib
from pykeen.pipeline import pipeline
from pyrdf2vec.samplers import ( 
    ObjFreqSampler,
    ObjPredFreqSampler,
    PageRankSampler,
    PredFreqSampler,
    UniformSampler
)

def kg_to_tsv(data):
    df = pd.DataFrame(columns=['subject','predicate','object'])
    g = Graph()
    g.parse(data)
    for s, p, o in g:
        if isinstance(o, Literal):
            continue
        df = pd.concat([df, pd.DataFrame([[s,p,o]], columns=['subject','predicate','object'])])
    df.to_csv(homedir +"/data/aifb_without_literals.tsv", sep="\t", index=False)
    return df

def create_rdf2vec_embedding(kg, entities):
    kg = KG(kg)
    transformer = RDF2VecTransformer(
        Word2Vec(epochs=10),
        walkers=[RandomWalker(4, 10, with_reverse=False, n_jobs=2)],
    )
    embeddings, literals = transformer.fit_transform(kg, entities)  # gesamter Graph
    emb_train = embeddings[:140]
    emb_test = embeddings[140:]
    with open(homedir + "/data/train_embedding", "wb") as fp:   #Pickling
        pickle.dump(emb_train, fp)
    with open(homedir + "/data/test_embedding", "wb") as fp:   #Pickling
        pickle.dump(emb_test, fp)
    return emb_train, emb_test

def create_pykeen_emb (traindata, testdata, entities):
    pipeline_result = pipeline(
    model='TransE',
    evaluation_entity_whitelist=entities,
    training=traindata,  # --> gesamten Datensatz verwenden, literals müssen raus 
    testing=testdata)
    pipeline_result.save_to_directory('../data/aifb_transE.pkl')
    return pipeline_result

def SVM_classifier(train_emb, test_emb, traindata, testdata):
    SVM_classifier = svm.SVC(kernel='linear', C=1.0, random_state=42)
    print(len(train_emb))
    print(traindata["label_affiliation"].shape)
    SVM_classifier.fit(train_emb, traindata["label_affiliation"])
    predictions = SVM_classifier.predict(test_emb)
    score = SVM_classifier.score(test_emb, testdata["label_affiliation"])
    return predictions, score


if __name__ == '__main__':
    homedir = "C:/Users/luisa/Projekte/Masterthesis/AIFB"
    traindata = pd.read_csv(homedir +"/data/trainingSet.tsv", sep="\t") # train und test zusammen
    testdata = pd.read_csv(homedir + "/data/testSet.tsv", sep="\t")
    kg = homedir +"/data/aifb_witho_complete.nt"
    #df = kg_to_tsv(kg)
    pykeen_data = homedir + '/data/trainingSet.tsv' #TriplesFactory.from_path(homedir + "/data/aifb_without_literals.tsv", sep="\t")
    pykeen_test = homedir + '/data/testSet.tsv' #TriplesFactory.from_path(homedir + "/data/testSet_pykeen.tsv", sep="\t")
    print(pykeen_data)
    data = pd.concat([traindata, testdata])
    entities = [entity for entity in traindata["person"]]
    pykeen_emb = create_pykeen_emb(pykeen_data, pykeen_test, entities)
    #train_emb, test_emb = create_rdf2vec_embedding(kg, entities)
    #pred, score = SVM_classifier(train_emb, test_emb, traindata, testdata)  
    #print(score)
    #print(pred)
    



    
    

