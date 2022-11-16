import os

import altair as alt

import pandas as pd
import numpy as np
import umap
from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from sentence_transformers import SentenceTransformer
from cleantext import clean
import nltk

import argparse


def clean_data(data: list, 
               save_dir: str):  
    """Cleans data by removing stopwords, punctuation, and numbers. 
    Splits the result into sentences. 
    Returns a list of cleaned sentences.

    Args:
        data (list): Text data stored in a list of strings.
        save_dir (str): The directory to save the cleaned data to.
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


def visualize(text_data: list, 
              embeddings: np.ndarray, 
              classes: list, 
              title: str, 
              save_dir: str):
    """Provides an interactive visualization of the clustered 2D UMAP embeddings and plots the clusters.
    
    Args:
        text_data (list): Text data stored in a list of strings.
        embeddings (np.ndarray): The 2D UMAP embeddings of the text data.
        classes (np.ndarray): The cluster labels of the embeddings.
        title (str): The title of the plot.
        save_dir (str): The directory to save the plot to.
    """
    # fix class labels
    class_names = ['-1_noise' if i == -1 else f'{i}_cluster' for i in classes]
    
    # collect data into data frame
    df = pd.concat([pd.DataFrame(data=text_data, columns=['Text']),  
                    pd.DataFrame(data=embeddings[:, 0], columns=['X']),
                    pd.DataFrame(data=embeddings[:, 1], columns=['Y']), 
                    pd.DataFrame(data=class_names, columns=['Cluster'])], axis=1)
    
    # create interactive chart for data
    chart = alt.Chart(df).mark_circle(size=150).encode(
        x=alt.X('X', scale=alt.Scale(zero=False), axis=alt.Axis(labels=False, ticks=False, domain=False)),
        y=alt.Y('Y', scale=alt.Scale(zero=False), axis=alt.Axis(labels=False, ticks=False, domain=False)),
        color=alt.Color('Cluster', scale=alt.Scale(scheme='tableau20')),
        tooltip=['Text']).properties(
            width=600, 
            height=600, 
            title=title).configure_legend(
                orient='right', 
                titleFontSize=16,
                labelFontSize=16,
                ).interactive()
            
    chart.save(save_dir + title + '_plot.html')
    chart.show()
    
    
def extract_topics(text_data: list, 
                   classes: list, 
                   apply_tfidf: bool, 
                   title: str,
                   save_dir: str):
    """Extracts the most common words in each cluster.
    
    Args:
        text_data (list): Text data stored in a list of strings.
        classes (np.ndarray): The cluster labels.
        apply_tfidf (bool): Whether to apply the TF-IDF to the within cluster word counts.
        title (str): The title of the csv file.
        save_dir (str): The directory to save the plot to.
    """
    unique_classes = np.unique(classes)
    
    # get bag of words for text data
    vectorizer = CountVectorizer(stop_words='english')
    text_bow = vectorizer.fit_transform(text_data).toarray() 
    
    # get vocabulary
    vocab = vectorizer.vocabulary_
    vocab = {v: k for k, v in vocab.items()}
    
    # get bag of words for clusters
    clusters_bow = [text_bow[classes == i].sum(axis=0) for i in unique_classes] 
    clusters_bow = np.stack(clusters_bow) 
    
    # apply class tf-idf
    if apply_tfidf:
        n_features = clusters_bow.shape[1]
        
        f_x = clusters_bow.sum(axis=0) # freq of words across all classes
        avg = clusters_bow.sum(axis=1).mean() # calculate avg number of words per cluster
        clusters_bow_norm = normalize(clusters_bow, axis=1, norm='l1', copy=False)
        
        idf = np.log((avg / f_x)+1)
        np.fill_diagonal(np.zeros((n_features, n_features)), idf)
        
        clusters_bow = clusters_bow_norm * idf 
        
    topics = []
    
    for i in range(clusters_bow.shape[0]):
        # get top 5 unigrams
        top5 = clusters_bow[i].argsort()[-5:] 
        top5 = [vocab[x] for x in top5]
        top5.reverse()
        top5 = '_'.join(top5)
        
        # get cluster size
        size = classes[classes == unique_classes[i]].shape[0]
        
        topics.append([unique_classes[i], size, top5])
        
    topics_df = pd.DataFrame(topics, columns=['class', 'size', 'topics'])
    topics_df.to_csv(path_or_buf=save_dir + title + "_topics.csv", index=False)
    
    return topics_df 


def run(data_path: str, 
        model: str, 
        embed_dim: int, 
        eps: float, 
        min_samples: int,
        apply_tfidf: bool,
        save_dir: str):
    """Runs the topic modeling pipeline.
    
    The pipeline consists of the following steps:
    - Load and clean the data.
    - Embed the data using a pretrained sentence transformer model.
    - Reduce the dimensionality of the embeddings using UMAP.
    - Cluster the embeddings using DBSCAN.
    - Extract topics for each cluster.
    - Visualize the clusters.
    
    Args:
        data_path (str): The path to the .xlsx data.
        model (str): The sentence transformer model to use.
        embed_dim (int): The dimension to reduce the embeddings to using UMAP before clustering.
        eps (float): The epsilon value to use for DBSCAN.
        min_samples (int): The minimum number of samples to use for DBSCAN.
        apply_tfidf (bool): Whether to apply class tf-idf when extracting topics.
        save_dir (str): The directory to save the results to.
    """
    
    dataset = pd.read_excel(data_path, header=0)

    for i in range(len(dataset.columns)): 
        
        question = dataset.columns[i]

        # clean data
        answers = dataset[dataset.columns[i]]
        answers = answers.dropna().tolist()
        answers = clean_data(answers, save_dir) # 500
        
        # embed data
        embedding_model = SentenceTransformer(model)
        embeddings = embedding_model.encode(answers) 
        
        # reduce dimensionality of embeddings
        reducer = umap.UMAP(n_components=embed_dim, metric='cosine', random_state=0)
        embeddings = reducer.fit_transform(embeddings) 
        
        # cluster embeddings
        clustering_model = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        # clustering_model = HDBSCAN(min_cluster_size=15, metric='euclidean')
        classes = clustering_model.fit_predict(embeddings)
        
        # extract topics
        topics = extract_topics(text_data=answers, 
                                classes=classes, 
                                apply_tfidf=apply_tfidf, 
                                title=f'question_{i+1}',
                                save_dir=save_dir)

        if embed_dim > 2:
            reducer = umap.UMAP(n_components=2, metric='euclidean', random_state=0)
            embeddings = reducer.fit_transform(embeddings)
    
        # visualization
        visualize(text_data=answers, 
                  embeddings=embeddings, 
                  classes=classes,
                  title=f'question_{i+1}', 
                  save_dir=save_dir)
        break


if __name__ == '__main__':
    # python run.py --data_path ./classroom_norms.xlsx --model all-MiniLM-L12-v2 --embed_dim 2 --eps 0.4 --min_samples 15 --save_dir ./example/
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', type=str, default='./example/classroom_norms.xlsx', 
                        help='Path to excel (.xlsx) file containing survey answers.')
    
    parser.add_argument('--model', type=str, default='all-MiniLM-L12-v2', 
                        help='The embedding model from sentence-transformers.')
    
    parser.add_argument('--embed_dim', type=int, default=2, 
                        help='The dimension to reduce the embeddings to for clustering.')
    
    parser.add_argument('--eps', type=float, default=0.35, 
                        help='The epsilon value to use for DBSCAN.')
    
    parser.add_argument('--min_samples', type=int, default=10, 
                        help='The minimum number of samples to use for DBSCAN.')
    
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
        apply_tfidf=True,
        save_dir=args.save_dir)