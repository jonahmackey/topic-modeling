import os

import pandas as pd
import pandas as pd
import numpy as np
import umap
import altair as alt
from sklearn.cluster import KMeans

from cleantext import clean
from sentence_transformers import SentenceTransformer
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams

from collections import Counter
import argparse

nltk.download('stopwords')
nltk.download('punkt')


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
        clean_text = clean_text.replace('..', '.')
        clean_text = clean_text.replace('  ', ' ')
        
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
            width=700, 
            height=700, 
            title=title).configure_legend(
                orient='right', 
                titleFontSize=16,
                labelFontSize=16).interactive()
    
    chart.show()
    

def run(data_path: str, model: str, n_clusters: int, save_dir: str):
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

            unigrams_frequencies = Counter([])
            bigrams_frequencies = Counter([])
            trigrams_frequencies = Counter([])
            
            for text in cluster:
                tokens = [re.sub(r'\W+', '', t) for t in text.split() if len(t) > 1 or t.isalnum()]
                tokens = [value for value in tokens if value not in stopwords.words('english')]
                
                unigrams = ngrams(tokens, 1)
                bigrams = ngrams(tokens, 2)
                trigrams = ngrams(tokens, 3)
                
                unigrams_frequencies += Counter(unigrams)
                bigrams_frequencies += Counter(bigrams)
                trigrams_frequencies += Counter(trigrams)
            
            unigrams_frequencies = unigrams_frequencies.most_common()
            bigrams_frequencies = bigrams_frequencies.most_common()
            trigrams_frequencies = trigrams_frequencies.most_common()
            
            with open(save_dir + f'unigrams_question{i+1}_cluster{j+1}.txt', 'w') as f_unigrams, open(save_dir + f'bigrams_question{i+1}_cluster{j+1}.txt', 'w') as f_bigrams, open(save_dir + f'trigrams_question{i+1}_cluster{j+1}.txt', 'w') as f_trigrams:
                f_unigrams.write(f'QUESTION: \n{question}\n\n')
                f_bigrams.write(f'QUESTION: \n{question}\n\n')
                f_trigrams.write(f'QUESTION: \n{question}\n\n')
                
                f_unigrams.write('\n'.join([' '.join(words) + f': {count}' for words, count in unigrams_frequencies]))
                f_bigrams.write('\n'.join([' '.join(words) + f': {count}' for words, count in bigrams_frequencies]))
                f_trigrams.write('\n'.join([' '.join(words) + f': {count}' for words, count in trigrams_frequencies]))
        
        visualize(answers=answers, 
                  embeddings=embeddings, 
                  classes=classes, 
                  title=f'Question {i+1}')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', type=str, default='./classroom_norms.xlsx', 
                        help='Path to excel (.xlsx) file containing survey answers.')
    
    parser.add_argument('--model', type=str, default='all-mpnet-base-v2', 
                        help='The embedding model from sentence-transformers.')
    
    parser.add_argument('--n_clusters', type=int, default=3, 
                        help='The number of clusters for kmeans algorithm.')
    
    parser.add_argument('--save_dir', type=str, default='./examples2/', 
                        help='The directory to save results in.')
    
    args = parser.parse_args()
    
    try:
        os.mkdir(args.save_dir)
    except:
        pass
    
    run(data_path=args.data_path,
        model=args.model,
        n_clusters=args.n_clusters,
        save_dir=args.save_dir)