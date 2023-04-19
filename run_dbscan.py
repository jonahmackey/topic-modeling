import os
import altair as alt
import altair_viewer
import pandas as pd
import numpy as np
import umap
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer

import argparse

from run import clean_data, extract_topics

def visualize(text_data: list, 
              embeddings: np.ndarray, 
              classes: list, 
              save_dir: str, 
              notebook: False):
    """Provides an interactive visualization of the clustered 2D UMAP embeddings and plots the clusters.
    
    Args:
        text_data (list): Text data stored in a list of strings.
        embeddings (np.ndarray): The 2D UMAP embeddings of the text data.
        classes (np.ndarray): The cluster labels of the embeddings.
        save_dir (str): The directory to save the plot to.
        notebook (bool): Whether to display the plot in a notebook.
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
            height=600).configure_legend(
                orient='right', 
                titleFontSize=16,
                labelFontSize=16,
                ).interactive()
            
    chart.save(save_dir + 'visual.html')
    
    if notebook:
        altair_viewer.display(chart, inline=True)
    else:
        altair_viewer.show(chart)


def cluster_dbscan(data_path: str,
                   model: str, 
                   sent: bool,
                   eps: float, 
                   min_samples: int,
                   save_dir: str,
                   notebook: bool):
    """Runs the first phase of the topic modeling pipeline.
    
    This function completes of the following steps:
    - Load and clean the text data.
    - Embed the data using a pre-trained sentence transformer model.
    - Reduce the dimensionality of the embeddings using UMAP.
    - Cluster the embeddings using DBSCAN.
    - Visualize the clusters.
    
    Args:
        data_path (str): The path to the .xlsx data.]
        model (str): The sentence transformer model to use.
        sent (bool): Whether to split the text into sentences.
        embed_dim (int): The dimension to reduce the embeddings to using UMAP before clustering.
        eps (float): The epsilon value to use for DBSCAN.
        min_samples (int): The minimum number of samples to use for DBSCAN.
        save_dir (str): The directory to save the results to.
        notebook (bool): Whether to display the clusters in a notebook.
    """
    
    dataset = pd.read_excel(data_path, header=0)    
    question = dataset.columns[0]
    
    print('QUESTION:\n    ', f'"{question}"')
    # clean data
    answers = dataset[dataset.columns[0]].dropna().tolist()
    answers = clean_data(answers, sent=sent) 
    
    # embed data
    embedding_model = SentenceTransformer(model)
    embeddings = embedding_model.encode(answers) 
    
    # reduce dimensionality of embeddings
    reducer = umap.UMAP(n_components=2, metric='cosine', random_state=0)
    embeddings = reducer.fit_transform(embeddings) 
    
    # cluster embeddings
    clustering_model = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    classes = clustering_model.fit_predict(embeddings)
    
    # visualization
    visualize(text_data=answers, 
              embeddings=embeddings, 
              classes=classes,
              save_dir=save_dir,
              notebook=notebook)
    
    return answers, classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', type=str, default='./datasets/expect_from_peers.xlsx', 
                        help='Path to excel (.xlsx) file containing survey answers.')
    
    parser.add_argument('--model', type=str, default='all-MiniLM-L12-v2', 
                        help='The embedding model from sentence-transformers.')
    
    parser.add_argument('--sent', type=str, default='true', 
                        help='Whether or not to split up the input text into sentences.')
    
    parser.add_argument('--eps', type=float, default=0.35, 
                        help='The epsilon value to use for DBSCAN.')
    
    parser.add_argument('--min_samples', type=int, default=10, 
                        help='The minimum number of samples to use for DBSCAN.')
    
    parser.add_argument('--ignore_words', nargs='+', default=['expect', 'peers'],
                        help='Words to ignore when extracting cluster topics.')
    
    parser.add_argument('--save_dir', type=str, default='./example2/', 
                        help='The directory to save results in.')
    
    args = parser.parse_args()
    
    try:
        os.mkdir(args.save_dir)
    except:
        pass
    
    sent = (args.sent == 'true')
    
    # clustering
    answers, classes = cluster_dbscan(data_path=args.data_path,
                                      model=args.model,
                                      sent=args.sent,
                                      eps=args.eps,
                                      min_samples=args.min_samples,
                                      save_dir=args.save_dir,
                                      notebook=False)
    
    # extract topics
    topics = extract_topics(text_data=answers, 
                            classes=classes, 
                            apply_tfidf=True, 
                            ignore_words=args.ignore_words,
                            save_dir=args.save_dir)
