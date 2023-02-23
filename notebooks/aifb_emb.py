import pandas as pd
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker
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
if __name__ == '__main__':

    homedir = "C:/Users/luisa/Projekte/Masterthesis/AIFB"
    # Read a CSV file containing the entities we want to classify.
    traindata = pd.read_csv(homedir +"/data/trainingSet.tsv", sep="\t") # train und test zusammen
    testdata = pd.read_csv(homedir + "/data/testSet.tsv", sep="\t")
    kg = KG(homedir +"/data/aifb_witho_complete.nt")
    data = pd.concat([traindata, testdata])
    entities = [entity for entity in data["person"]]
    transformer = RDF2VecTransformer(
    Word2Vec(epochs=10),
    walkers=[RandomWalker(4, 10, with_reverse=False, n_jobs=2)],
    # verbose=1
    )
    # Get our embeddings.
    embeddings, literals = transformer.fit_transform(kg, entities)  # gesamter Graph
    print(embeddings) # output fit transform um test und train embedding zu erhalten
    print(len(embeddings))
    emb_train = embeddings[:140]
    emb_test = embeddings[140:]
    print(emb_test)
    with open(homedir + "/data/train_embedding", "wb") as fp:   #Pickling
        pickle.dump(emb_train, fp)
    with open(homedir + "/data/test_embedding", "wb") as fp:   #Pickling
        pickle.dump(emb_test, fp)