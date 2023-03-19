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
from pykeen.pipeline import pipeline

homedir = "C:/Users/luisa/Projekte/Masterthesis/AIFB"
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
        Word2Vec(epochs=250), # mehr Epochen (100/200), mehr als 75% accuracy
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

def save_rdf2vec_emb(emb_train, emb_test):
    with open(homedir + "/data/train_rdf_embedding", "wb") as fp:   #Pickling
        pickle.dump(emb_train, fp)
    with open(homedir + "/data/test_rdf_embedding", "wb") as fp:   #Pickling
        pickle.dump(emb_test, fp)

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
    embeddings_np = np.array(embeddings)


    pykeen_emb_train = embeddings_np[:140]
    pykeen_emb_test = embeddings_np[140:]
    return pykeen_emb_train, pykeen_emb_test

def save_pykeen_emb (emb_train, emb_test):
    with open(homedir + "/data/train_pykeen_embedding", "wb") as fp:   #Pickling
        pickle.dump(emb_train, fp)
    with open(homedir + "/data/test_pykeen_embedding", "wb") as fp:   #Pickling
        pickle.dump(emb_test, fp)





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
    pykeen_emb_train, pykeen_emb_test = create_pykeen_embedding(pykeen_data, pykeen_test, entities)
    train_emb, test_emb = create_rdf2vec_embedding(kg, entities)
    save_pykeen_emb(pykeen_emb_train, pykeen_emb_test)
    save_rdf2vec_emb(train_emb, test_emb)
    pred_rdf_G, score_rdf_G = Gaussian_classifier(train_emb, test_emb, traindata, testdata)
    pred_rdf_py, score_py_G = Gaussian_classifier(pykeen_emb_train, pykeen_emb_test, traindata, testdata) # size:140x50
    pred_rdf_SVM, score_rdf_SVM = SVM_classifier(train_emb, test_emb, traindata, testdata)
    pred_py_SVM, score_py_SVM = SVM_classifier(pykeen_emb_train, pykeen_emb_test, traindata, testdata)
    #np.savetxt(homedir + "/data/results/prediction_Gaussianclassifier.txt", pred,fmt="%s")
    print('Score_rdf_Gaussian Kernel: ', score_rdf_G)
    print('Score_pykeen_Gaussian Kernel: ', score_py_G)
    print('Score_rdf_SVM: ', score_rdf_SVM)
    print('Score_pykeen_SVM: ', score_py_SVM)
    



    
    

