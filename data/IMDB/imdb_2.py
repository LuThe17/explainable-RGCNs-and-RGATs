from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import os
import gzip
import json
import tqdm
import imdb
import pandas as pd
from rdflib import Graph, URIRef, RDF, Literal, Namespace
from rdflib.namespace import RDF

ia = imdb.IMDb(accessSystem = 'http', reraiseExceotions=True)
homedir='/home/luitheob/AIFB/'

movies = pd.read_table(homedir + 'data/IMDB/entity2id.txt', delim_whitespace=True)#, names=('A', 'B', 'C'))



ex = Namespace('http://imdb.org/')
ont = Namespace('http://imdb.org/ontology/')
#title_uri = URIRef(ont+'has_title')
has_genre_uri = URIRef(ont+'has_genre')
director_uri = URIRef(ont+'has_director')
dir_mov_uri = URIRef(ont+'directed_by')
cast_uri = URIRef(ont+'has_actor')
cast_mov_uri = URIRef(ont+'acted_in')
published_uri = URIRef(ont+'published_in')
collab_uri = URIRef(ont+'collabs_with')
prefers_genre = URIRef(ont+'preferred_by')

dict = {}
test_train = []
rdf_graph = Graph()
# Genres to include
included_genres = ['Drama', 'Comedy', 'Romance', 'Thriller']
id = 0 

# Define a function to process a single movie
def process_movie(movie_id, included_genres,rdf_graph, id):
    try:
        movie = ia.get_movie(movie_id)  # Remove 'tt' from the ID
        #print(movie)
        id +=1
        # Check if the movie's genre is in the included genres list
        movie_genres = movie.get('genres', [])
        #print(movie_genres)
        #print(included_genres)
        if any(genre in included_genres for genre in movie_genres):
            # Create URIs for movie and its attributes
            movie_uri = URIRef(ex+'mov'+movie_id)
            dict[movie_id] = movie.get('title')
            # Add triples for movie attributes
           # rdf_graph.add((movie_uri, title_uri, Literal(movie_id)))
            year_uri = URIRef(ex+'year')
            rdf_graph.add((movie_uri, published_uri, year_uri))
        
            
            # Add genre triples
            for genre in movie_genres[0]:
                if genre in included_genres:
                    genre_uri = URIRef(ex+'gen'+genre)
                    #rdf_graph.add((movie_uri, has_genre_uri, genre_uri))
            test_train.append([movie_uri, id, genre_uri])
            # Add director triples
            for director in movie.get('director', []):
                director_name = director.get('name')
                director_id = director.getID()
                dict[director_id] = director_name
                dir_uri = URIRef(ex+'dir'+director_id)
                rdf_graph.add((movie_uri, director_uri, dir_uri))
                rdf_graph.add((dir_uri, dir_mov_uri, movie_uri))
            
            # Add cast triples
            for actor in movie.get('cast', []):
                actor_name = actor.get('name')
                actor_id = actor.getID()
                dict[actor_id] = actor_name
                act_uri = URIRef(ex+'act'+actor_id)
                rdf_graph.add((movie_uri, cast_uri, act_uri))
                rdf_graph.add((act_uri, cast_mov_uri, movie_uri))
                rdf_graph.add((act_uri, collab_uri, dir_uri))
                rdf_graph.add((dir_uri, collab_uri, act_uri))
                rdf_graph.add((genre_uri, prefers_genre, act_uri))
                rdf_graph.add((genre_uri, prefers_genre, dir_uri))
            
    except imdb.IMDbError:
        print(IMDbError)
        pass

# List of movie IDs
movie_ids = movies['movie'][:1500]
# Create an RDF graph
rdf_graph = Graph()

# Number of concurrent threads/processes
num_threads = 5  # Adjust as needed

# Create a ThreadPoolExecutor
# Create a ProcessPoolExecutor
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(process_movie, movie_id[2:], included_genres,rdf_graph, idx + 1) for idx, movie_id in enumerate(movie_ids)]
    for future in concurrent.futures.as_completed(futures):
        pass 
# Serialize the RDF graph to a file (e.g., Turtle format)
rdf_graph.serialize(homedir +'data/IMDB/imdb.nt', format='nt')

train = pd.DataFrame(test_train, columns=['movie', 'id','genre'])

from sklearn.model_selection import train_test_split
def stratified_train_test_split(train_test):
    train, test = train_test_split(train_test, test_size=0.06, random_state=42, stratify=train_test['genre'])
    return train, test
train, test = stratified_train_test_split(train)
test
train.to_csv(homedir+'data/IMDB/training_Set.csv', index=False)
test.to_csv(homedir + 'data/IMDB/test_Set.csv', index=False)

tsv_filename = 'movie_kg.tsv'
with open(tsv_filename, 'w') as tsv_file:
    for s, p, o in rdf_graph:
        tsv_file.write(f'{s}\t{p}\t{o}\n')
