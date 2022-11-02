import os

import pandas as pd
import numpy as np
import umap
import altair as alt
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans 
from sklearn.decomposition import PCA

from sentence_transformers import SentenceTransformer

from cleantext import clean
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams

from collections import Counter
import argparse

nltk.download('stopwords')
nltk.download('punkt')


def clean_data(data: list, 
               save_dir: str):  
    """Cleans data by removing stopwords, punctuation, and numbers. 
    Splits the result into sentences. 
    Returns a list of cleaned sentences.

    Args:
        data (list): Text data stored in a list of strings.
        save_dir (str): The directory to save the cleaned data to.

    Returns:
        list: The cleaned data stored in a list of strings.
    """
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


def plot_ngrams(ngrams: list, 
                topn: int, 
                title: str, 
                save_path: str):
    """Plots the top ngrams in the data.
    
    Args:
        ngrams (int): The ngrams to plot.
        topn (int): The number of ngrams to plot.
        title (str): The title of the plot.
        save_path (str): The path to save the plot to.
    """
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


def visualize(text_data: list, 
              embeddings: np.ndarray, 
              classes: list, 
              title: str, 
              save_dir: str):
    """Provides an interactive visualization of the clustered 2D UMAP embeddings and plots the clusters.
    
    Args:
        data (list): Text data stored in a list of strings.
        embeddings (np.ndarray): The 2D UMAP embeddings of the text data.
        classes (list): The cluster labels of the embeddings.
        title (str): The title of the plot.
        save_dir (str): The directory to save the plot to.
    """
    # fix class labels
    class_names = []
    
    for i in classes:
        if i == -1:
            name = 'noise'
        else:
            name = f'cluster {i+1}'
            
        class_names.append(name)

    # collect data into data frame
    df = pd.concat([pd.DataFrame(data=text_data, columns=['Answer']),  
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
    

def run(data_path: str, 
        model: str, 
        embed_dim: int, 
        eps: float, 
        min_samples: int, 
        save_dir: str):
    """Runs the topic modeling pipeline.
    
    The pipeline consists of the following steps:
    - Load and clean the data.
    - Embed the data using a pretrained sentence transformer model.
    - Reduce the dimensionality of the embeddings using UMAP.
    - Cluster the embeddings using DBSCAN.
    - Visualize the clusters and plot the top ngrams in each cluster.
    
    Args:
        data_path (str): The path to the .xlsx data.
        model (str): The sentence transformer model to use.
        embed_dim (int): The dimension to reduce the embeddings to using UMAP before clustering.
        eps (float): The epsilon value to use for DBSCAN.
        min_samples (int): The minimum number of samples to use for DBSCAN.
        save_dir (str): The directory to save the results to.
    """
    
    dataset = pd.read_excel(data_path, header=0)

    for i in range(len(dataset.columns)): 
        
        question = dataset.columns[i]

        # clean data
        answers = dataset[dataset.columns[i]]
        answers = answers.dropna().tolist()
        answers = clean_data(answers, save_dir) 
        
        # embed data
        embedding_model = SentenceTransformer(model)
        embeddings = embedding_model.encode(answers) # 384 dims
        
        # reduce dimensionality of embeddings
        reducer = umap.UMAP(n_components=embed_dim, metric='cosine', random_state=0)
        embeddings = reducer.fit_transform(embeddings)
        
        # cluster embeddings
        clustering_model = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        classes = clustering_model.fit_predict(embeddings).tolist()
        
        # plot ngrams for each cluster
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
            plot_ngrams(bigrams_frequencies, 20, f'Bigrams (question {i+1}, cluster {j+1})', save_dir + f'question{i+1}_cluster{j+1}_bigrams.png') 

        if embed_dim > 2:
            reducer = umap.UMAP(n_components=2, metric='euclidean', random_state=0)
            embeddings = reducer.fit_transform(embeddings)
        
        # visualization
        visualize(text_data=answers, 
                  embeddings=embeddings, 
                  classes=classes, 
                  title=f'Question {i+1}', 
                  save_dir=save_dir)


if __name__ == '__main__':
    
    # python run.py --data_path ./classroom_norms.xlsx --model all-MiniLM-L12-v2 --embed_dim 2 --eps 0.4 --min_samples 15 --save_dir ./example/
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', type=str, default='./classroom_norms.xlsx', 
                        help='Path to excel (.xlsx) file containing survey answers.')
    
    parser.add_argument('--model', type=str, default='all-MiniLM-L12-v2', 
                        help='The embedding model from sentence-transformers.')
    
    parser.add_argument('--embed_dim', type=int, default=2, 
                        help='The dimension to reduce the embeddings to for clustering.')
    
    parser.add_argument('--eps', type=float, default=0.35, 
                        help='The epsilon value to use for DBSCAN.')
    
    parser.add_argument('--min_samples', type=int, default=10, 
                        help='The minimum number of samples to use for DBSCAN.')
    
    parser.add_argument('--save_dir', type=str, default='./example2/', 
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