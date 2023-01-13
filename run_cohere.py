import os

import pandas as pd
import umap
from sklearn.cluster import DBSCAN
import cohere

from run import clean_data, visualize, extract_topics

import argparse


def cluster_cohere(data_path: str,
                   question_n: int,
                   model_size: str,
                   api_key: str, 
                   embed_dim: int, 
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
        data_path (str): The path to the .xlsx data.
        question_n (int): The question number to analyze from the survey.
        model_size (str): The size of embedding model to use. Options: small, medium, large.
        api_key (str): The API key to use for the Cohere API.
        embed_dim (int): The dimension to reduce the embeddings to using UMAP before clustering.
        eps (float): The epsilon value to use for DBSCAN.
        min_samples (int): The minimum number of samples to use for DBSCAN.
        save_dir (str): The directory to save the results to.
        notebook (bool): Whether to display the clusters in a notebook.
    """
    
    dataset = pd.read_excel(data_path, header=0)    
    question = dataset.columns[question_n]
    
    print('QUESTION:\n    ', f'"{question}"')

    # clean data
    answers = dataset[dataset.columns[question_n]].dropna().tolist()
    answers = clean_data(answers) 
    
    # embed data
    co = cohere.Client(api_key) 
    response = co.embed(model=model_size, texts=answers) 
    embeddings = response.embeddings 
    
    # reduce dimensionality of embeddings
    reducer = umap.UMAP(n_components=embed_dim, metric='cosine', random_state=0)
    embeddings = reducer.fit_transform(embeddings) 
    
    # cluster embeddings
    clustering_model = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    classes = clustering_model.fit_predict(embeddings)

    # visualization
    if embed_dim > 2:
        reducer = umap.UMAP(n_components=2, metric='euclidean', random_state=0)
        embeddings = reducer.fit_transform(embeddings)

    visualize(text_data=answers, 
              embeddings=embeddings, 
              classes=classes,
              save_dir=save_dir,
              notebook=notebook)
    
    return answers, classes

if __name__=="__main__":
    # python run.py --data_path ./classroom_norms.xlsx --model_size small --embed_dim 2 --eps 0.4 --min_samples 15 --api_key '' --save_dir ./example/
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', type=str, default='./example/classroom_norms.xlsx', 
                        help='Path to excel (.xlsx) file containing survey answers.')
    
    parser.add_argument('--question_n', type=int, default=0, 
                        help='Question number to analyze from the survey.')
    
    parser.add_argument('--model_size', type=str, default='small', 
                        help='The size of embedding model to use. Options: small and large.')
    
    parser.add_argument('--embed_dim', type=int, default=2, 
                        help='The dimension to reduce the embeddings to for clustering.')
    
    parser.add_argument('--eps', type=float, default=0.35, 
                        help='The epsilon value to use for DBSCAN.')
    
    parser.add_argument('--min_samples', type=int, default=10, 
                        help='The minimum number of samples to use for DBSCAN.')
    
    parser.add_argument('--api_key', type=str, default='YOUR_API_KEY',
                        help='api_key (str): The API key to use for the Cohere API.')
    
    parser.add_argument('--save_dir', type=str, default='./example/', 
                        help='The directory to save results in.')
    
    args = parser.parse_args()
    
    try:
        os.mkdir(args.save_dir)
    except:
        pass
    
    answers, classes = cluster_cohere(data_path=args.data_path,
                            question_n=args.question_n,
                            model_size=args.model_size,
                            api_key=args.api_key,
                            embed_dim=args.embed_dim,
                            eps=args.eps,
                            min_samples=args.min_samples,
                            save_dir=args.save_dir,
                            notebook=False)
    
    topics = extract_topics(text_data=answers, 
                            classes=classes, 
                            apply_tfidf=True, 
                            ignore_words=['expect'],
                            save_dir=args.save_dir)