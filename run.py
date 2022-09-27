import os

import umap
import altair as alt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import argparse

from sklearn.cluster import KMeans
from cleantext import clean
from sentence_transformers import SentenceTransformer
import altair as alt

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.util import ngrams

from collections import Counter


def clean_data(data: list, view: bool, save_dir: str):
    cleaned_data = []
    
    if view:
        txt_file_str = ''
        
    for i in range(len(data)):
        text = data[i]
        clean_text = clean(text)
        clean_text = clean_text.replace('*', '')
        clean_text = clean_text.replace('-', '')
        clean_text = clean_text.replace('\n', '. ')
        
        if view:
            txt_file_str += f'BEFORE: \n\n{text} \n\nAFTER: \n\n{clean_text} \n\n' + 80 * '-' + '\n\n'
        
        cleaned_data.append(clean_text)
        
    if view:
        with open(save_dir + 'data_cleaning.txt', 'w') as f:
            f.write(txt_file_str)
    
    return cleaned_data


def visualize(answers, embeddings, classes, title):
    # reduce dimension of embeddings to 2d using UMAP
    reducer = umap.UMAP()
    reduced_embeddings = reducer.fit_transform(embeddings)

    # fix class labels
    classes = [f'cluster {i+1}' for i in classes]

    # collect data into data frame
    df = pd.concat([pd.DataFrame(data=answers, columns=['Answer']),  
                    pd.DataFrame(data=reduced_embeddings[:, 0], columns=['X']),
                    pd.DataFrame(data=reduced_embeddings[:, 1], columns=['Y']), 
                    pd.DataFrame(data=classes, columns=['Cluster'])], axis=1)
    
    # create interactive chart for data
    chart = alt.Chart(df).mark_circle(size=150).encode(
        x=alt.X('X', scale=alt.Scale(zero=False), axis=alt.Axis(labels=False, ticks=False, domain=False)),
        y=alt.Y('Y', scale=alt.Scale(zero=False), axis=alt.Axis(labels=False, ticks=False, domain=False)),
        color='Cluster',
        tooltip=['Answer']).properties(
            width=500, 
            height=500, 
            title=title).configure_legend(
                orient='right', 
                titleFontSize=16,
                labelFontSize=16).interactive()
    
    chart.show()
    

def run(data_path: str, model: str, n_clusters: int, n_grams: int, save_dir: str):
    dataset = pd.read_excel(data_path, header=0)

    for i in range(len(dataset.columns)):
        
        # get survey question
        question = dataset.columns[i]
        
        print(f'Processing question: \n    {question}')

        # get survey answers
        answers = dataset[dataset.columns[i]]
        answers = answers.dropna().tolist() # remove nan entries
        answers = clean_data(answers, True, save_dir) # clean data
        
        # embed text
        embedding_model = SentenceTransformer(model)
        embeddings = embedding_model.encode(answers) 
        
        # kmeans on embedded text
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=0)
        classes = kmeans_model.fit_predict(embeddings).tolist()
        
        # get top ngrams for each cluster
        for j in range(n_clusters):
            cluster = np.array(answers)[np.array(classes) == j]

            frequencies = Counter([])
            
            for text in cluster:
                tokens = [re.sub(r'\W+', '', t) for t in text.split() if len(t) > 1 or t.isalnum()]
                tokens = [value for value in tokens if value not in stopwords.words('english')]
                
                grams = ngrams(tokens, n_grams)
                frequencies += Counter(grams)
            
            frequencies = frequencies.most_common()
            
            with open(save_dir + f'ngrams_question{i}_cluster{j}.txt', 'w') as f:
                f.write(f'QUESTION: \n{question}\n\n')
                
                for words, count in frequencies:
                    f.write(' '.join(words) + f': {count}\n')
        
        visualize(answers=answers, 
                  embeddings=embeddings, 
                  classes=classes, 
                  title=question)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', type=str, default='./example/classroom_norms.xlsx', 
                        help='Path to excel (.xlsx) file containing survey answers.')
    
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2', 
                        help='The embedding model from sentence-transformers.')
    
    parser.add_argument('--n_clusters', type=int, default=3, 
                        help='The number of clusters for kmeans algorithm.')
    
    parser.add_argument('--n_grams', type=int, default=2, 
                        help='The number of clusters for kmeans algorithm.')
    
    parser.add_argument('--save_dir', type=str, default='./example/', 
                        help='The directory to save results in.')
    
    args = parser.parse_args()
    
    try:
        os.mkdir(args.save_dir)
    except:
        pass
    
    run(data_path=args.data_path,
        model=args.model,
        n_clusters=args.n_clusters, 
        n_grams=args.n_grams,
        save_dir=args.save_dir)