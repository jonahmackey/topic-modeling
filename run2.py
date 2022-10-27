import os

import pandas as pd
import pandas as pd
import numpy as np
import umap
import altair as alt
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from sentence_transformers import SentenceTransformer
from transformers import pipeline

from cleantext import clean
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams

from collections import Counter
import argparse

summarizer = pipeline("summarization")

nltk.download('stopwords')
nltk.download('punkt')


def clean_data(data: list, save_dir: str):
    cleaned_data = []
    txt_file_str = ''

    for i in range(len(data)):
        text = data[i]
        clean_text = clean(text)
        clean_text = clean_text.replace('*', '')
        clean_text = clean_text.replace('-', '')
        clean_text = clean_text.replace('\n', '. ')
        clean_text = clean_text.replace('..', '.')
        clean_text = clean_text.strip()
        clean_text = clean_text.replace('  ', ' ')
        if clean_text[0] == ' ':
            clean_text = clean_text[1:]
        if clean_text[-1] != '.':
            clean_text = clean_text + '.'
        clean_text = nltk.tokenize.sent_tokenize(clean_text)
        
        clean_text_lines = '\n'.join(clean_text)
        txt_file_str += f'BEFORE: \n\n{text} \n\nAFTER: \n\n{clean_text_lines} \n\n' + 80 * '-' + '\n\n'
        
        cleaned_data += clean_text
    
    with open(save_dir + 'data_cleaning.txt', 'w') as f:
        f.write(txt_file_str)
    
    return cleaned_data


def plot_ngrams(ngrams, topn, title, save_path):
    words = [' '.join(x) for x, c in ngrams][:topn]
    words.reverse()

    counts = [c for x, c in ngrams][:topn]
    counts.reverse()
    
    plt.figure()
    
    plt.barh(words, counts)
    plt.title(title)
    plt.xlabel('# of Occuurences')
    plt.tick_params(axis='y', labelsize=6, labelrotation=45)
    
    plt.savefig(save_path)


def visualize(answers, embeddings, classes, title, save_dir):
    
    # fix class labels
    class_names = []
    
    for i in classes:
        if i == -1:
            name = 'noise'
            
        else:
            name = f'cluster {i+1}'
            
        class_names.append(name)

    # collect data into data frame
    df = pd.concat([pd.DataFrame(data=answers, columns=['Answer']),  
                    pd.DataFrame(data=embeddings[:, 0], columns=['X']),
                    pd.DataFrame(data=embeddings[:, 1], columns=['Y']), 
                    pd.DataFrame(data=class_names, columns=['Cluster'])], axis=1)
    
    # create interactive chart for data
    chart = alt.Chart(df).mark_circle(size=150).encode(
        x=alt.X('X', scale=alt.Scale(zero=False), axis=alt.Axis(labels=False, ticks=False, domain=False)),
        y=alt.Y('Y', scale=alt.Scale(zero=False), axis=alt.Axis(labels=False, ticks=False, domain=False)),
        color=alt.Color('Cluster', scale=alt.Scale(scheme='tableau20')),
        tooltip=['Answer']).properties(
            width=700, 
            height=700, 
            title=title).configure_legend(
                orient='right', 
                titleFontSize=16,
                labelFontSize=16).interactive()
            
    chart.save(save_dir + title.lower().replace(" ", "_") + '_plot.html')
    chart.show()
    

def run(data_path, model, embed_dim, eps, min_samples, save_dir):
    dataset = pd.read_excel(data_path, header=0)

    for i in range(len(dataset.columns)):
        
        # get survey question
        question = dataset.columns[i]

        # get survey answers
        answers = dataset[dataset.columns[i]]
        answers = answers.dropna().tolist() # remove nan entries
        answers = clean_data(answers, save_dir) # clean data
         
        # embed text
        embedding_model = SentenceTransformer(model)
        embeddings = embedding_model.encode(answers) 
        
        reducer = umap.UMAP(n_components=embed_dim, metric='cosine', random_state=0)
        embeddings = reducer.fit_transform(embeddings)
        
        # kmeans on embedded text
        clustering_model = DBSCAN(eps=eps, min_samples=min_samples)
        classes = clustering_model.fit_predict(embeddings).tolist()
        
        # get top ngrams for each cluster
        for j in set(classes):
            cluster = np.array(answers)[np.array(classes) == j]

            unigrams_frequencies = Counter([])
            bigrams_frequencies = Counter([])
            
            for text in cluster:
                tokens = [re.sub(r'\W+', '', t) for t in text.split() if len(t) > 1 or t.isalnum()]
                tokens = [value for value in tokens if value not in stopwords.words('english')]
                
                unigrams = ngrams(tokens, 1)
                bigrams = ngrams(tokens, 2)
                
                unigrams_frequencies += Counter(unigrams)
                bigrams_frequencies += Counter(bigrams)
            
            unigrams_frequencies = unigrams_frequencies.most_common()
            bigrams_frequencies = bigrams_frequencies.most_common()
            
            plot_ngrams(unigrams_frequencies, 20, f'Unigrams (question {i+1}, cluster {j+1})', save_dir + f'question{i+1}_cluster{j+1}_unigrams.png')
            plot_ngrams(bigrams_frequencies, 20, f'Bigrams (question {i+1}, cluster {j+1})', save_dir + f'question{i+1}_cluster{j+1}_biigrams.png') 
        
        if embed_dim != 2:
            reducer = umap.UMAP(n_components=2, metric='euclidean', random_state=0)
            embeddings = reducer.fit_transform(embeddings)
        
        visualize(answers=answers, 
                  embeddings=embeddings, 
                  classes=classes, 
                  title=f'Question {i+1}', 
                  save_dir=save_dir)
        break


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', type=str, default='./classroom_norms.xlsx', 
                        help='Path to excel (.xlsx) file containing survey answers.')
    
    parser.add_argument('--model', type=str, default='all-mpnet-base-v2', 
                        help='The embedding model from sentence-transformers.')
    
    parser.add_argument('--embed_dim', type=int, default=15, 
                        help='The dimension to reduce the embeddings to for clustering.')
    
    parser.add_argument('--eps', type=float, default=0.7, 
                        help='The maximum distance between two samples for one to be considered as in the neighborhood of the other in the DBSCAN clustering algorithm.')
    
    parser.add_argument('--min_samples', type=int, default=15, 
                        help='The number of samples in a neighborhood for a point to be considered as a core point in the DBSCAN clustering algorithm.')
    
    parser.add_argument('--save_dir', type=str, default='./example/', 
                        help='The directory to save results in.')
    
    args = parser.parse_args()
    
    try:
        os.mkdir(args.save_dir)
    except:
        pass
    
    run(data_path=args.data_path,
        model=args.model,
        embed_dim=args.embed_dim,
        eps=args.eps,
        min_samples=args.min_samples,
        save_dir=args.save_dir)