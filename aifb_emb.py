import pandas as pd
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker
from pykeen.triples import TriplesFactory
from rdflib import Graph, Literal
import numpy as np
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import pickle
import rdflib
from pykeen.pipeline import pipeline


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
    print(embeddings)
    emb_train = embeddings[:140]
    emb_test = embeddings[140:]
    with open(homedir + "/data/train_embedding", "wb") as fp:   #Pickling
        pickle.dump(emb_train, fp)
    with open(homedir + "/data/test_embedding", "wb") as fp:   #Pickling
        pickle.dump(emb_test, fp)
    return emb_train, emb_test

def create_pykeen_embedding(train, test):
    
    results = pipeline(
        training=train,
        testing=test,
        model="TransE",
        #training_kwargs=dict(num_epochs=100),
        #random_seed=1235,
        device="cpu",
    )
    results.save_to_directory(homedir + "/data/pykeen_emb")
    print(results.metric_results.to_df())
    return results


    # triples_factory = pykeen.triples.TriplesFactory.from_path(kg)

    # # Filter the triples to include only those that involve the selected entities

    # filtered_triples = []
    # for triple in triples_factory.mapped_triples:
    #     if triple[0] in entities or triple[2] in entities:
    #         filtered_triples.append((triples_factory.entity_to_id[triple[0]], 
    #                                  triples_factory.relation_to_id[triple[1]], 
    #                                  triples_factory.entity_to_id[triple[2]]))

    
    # filtered_triples_factory = TriplesFactory.from_labeled_triples(
    #     triples=filtered_triples,
    #     entity_to_id=triples_factory.entity_to_id,
    #     relation_to_id=triples_factory.relation_to_id,
    # )

    # # Define the embedding model
    # model = pykeen.models.TransE(
    #     triples_factory=filtered_triples_factory)

    # # Train the model
    # result = model.fit(
    #     triples_factory=filtered_triples_factory)
    # print(result)
    # return result
    # pipeline_result = pipeline(
    # model='TransE',
    # evaluation_entity_whitelist=entities,
    # training=traindata,  # --> gesamten Datensatz verwenden, literals müssen raus 
    # testing=testdata)
    
    # return pipeline_result

def SVM_classifier(train_emb, test_emb, traindata, testdata):
    SVM_classifier = svm.SVC(kernel='linear', C=1.0, random_state=42)
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


if __name__ == '__main__':
    homedir = "C:/Users/luisa/Projekte/Masterthesis/AIFB"
    traindata = pd.read_csv(homedir +"/data/trainingSet.tsv", sep="\t") # train und test zusammen
    testdata = pd.read_csv(homedir + "/data/testSet.tsv", sep="\t")
    testpy = testdata[:2]
    kg = homedir +"/data/aifb_witho_complete.nt"
    #df = kg_to_tsv(kg)
    pykeen_data = TriplesFactory.from_path(homedir + "/data/aifb_without_literals.tsv", sep="\t")
    pykeen_test = TriplesFactory.from_path(homedir + "/data/testSetpy.tsv", sep="\t")
    data = pd.concat([traindata, testdata])
    entities = [entity for entity in data["person"]]
    #pykeen_emb = create_pykeen_embedding(pykeen_data, pykeen_test)
    train_emb, test_emb = create_rdf2vec_embedding(kg, entities)
    pred, score = Gaussian_classifier(train_emb, test_emb, traindata, testdata)  
    np.savetxt(homedir + "/data/results/prediction_Gaussianclassifier.txt", pred,fmt="%s")
    print(score)
    print(pred)
    



    
    

