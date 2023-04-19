import os

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from sentence_transformers import SentenceTransformer
from cleantext import clean
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

import argparse


def clean_data(text_data: list, sent: bool):  
    """Cleans the text_data and splits it up into sentences.

    Args:
        text_data (list): Text data stored in a list of strings.
    """
    cleaned_data = []

    for i in range(len(text_data)): 
        text = text_data[i]
        clean_text = clean(text)
        clean_text = clean_text.replace('*', '')
        clean_text = clean_text.replace('-', '')
        clean_text = clean_text.replace('\n', '. ')
        clean_text = clean_text.replace('..', '.')
        clean_text = clean_text.strip()
        clean_text = clean_text.replace('  ', ' ')
        if clean_text == '':
            continue
        if clean_text[0] == ' ':
            clean_text = clean_text[1:]
        if clean_text[-1] != '.':
            clean_text = clean_text + '.'
            
        if sent:
            clean_text = nltk.tokenize.sent_tokenize(clean_text)        
        else: 
            clean_text = [clean_text]
        
        cleaned_data += clean_text
    
    return cleaned_data

    
def extract_topics(text_data: list, 
                   classes: list, 
                   apply_tfidf: bool, 
                   ignore_words: list):
    """Extracts the top 5 unigrams from each cluster.
    
    Args:
        text_data (list): Text data stored in a list of strings.
        classes (np.ndarray): The cluster labels.
        apply_tfidf (bool): Whether to apply the TF-IDF to the within cluster word counts.
        ignore_words (list): A list of words to ignore when extracting the topics.
        save_dir (str): The directory to save the plot to.
    """
    stop_words = stopwords.words('english') + ignore_words
    
    unique_classes = np.unique(classes)
    
    # get bag of words for text data
    vectorizer = CountVectorizer(stop_words=stop_words)
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
        
        topics.append(top5)
    
    return topics


def cluster(data_path: str,
            model: str,  
            sent: bool,
            thresh: float):
    """Embeds and clusters the text data.
    
    Args:
        data_path (str): The path to the .xlsx data.
        model (str): The sentence transformer model to use.
        sent (bool): Whether to split the text into sentences.
        thresh (float): The threshold to use for agglomerative clustering.
    """
    
    dataset = pd.read_excel(data_path, header=0)    
    question = dataset.columns[0]
    
    print('QUESTION:\n    ', f'"{question}"')

    # clean data
    responses = dataset[dataset.columns[0]].dropna().tolist()
    responses = clean_data(responses, sent=sent) 

    # embed data
    embedding_model = SentenceTransformer(model)
    embeddings = embedding_model.encode(responses) 
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # cluster embeddings
    clustering_model = AgglomerativeClustering(n_clusters=None, 
                                               distance_threshold=thresh, 
                                               metric='euclidean',
                                               linkage='ward')
    classes = clustering_model.fit_predict(embeddings) # (556,)
    
    return responses, classes, embeddings

def save_results(responses: list, 
                 classes: list, 
                 embeddings: np.ndarray,
                 topics: list,
                 save_dir: str):
    """Saves results to a .txt file. For each cluster, results are saved in the following format:
    
    CLUSTER <cluster_id> | SIZE <cluster_size> | TOPIC <cluster_topics>:
        <response_1> - <response_5>
        ...
        <response_n-5> - <response_n>
    
    where the responses are sorted by distance from the cluster mean.
    
    Args:
        responses (list): The list of responses.
        classes (np.ndarray): The cluster labels.
        embeddings (np.ndarray): The text embeddings.
        topics (list): The topics for each cluster.
        save_dir (str): The directory to save the results to.
    """
    responses = np.array(responses)
    classes = np.array(classes)
    topics = np.array(topics)
    
    # sort clusters by size
    unique_classes = np.unique(classes)
    cluster_sizes = np.array([len(embeddings[classes == i]) for i in unique_classes])
    cluster_sorted = np.flip(np.argsort(cluster_sizes)) # shape? 
    
    result = ''
    for i in range(len(unique_classes)):
        cluster_idx = cluster_sorted[i]
        
        # get cluster and responses
        cluster = embeddings[classes == cluster_idx]
        cluster_responses = responses[classes == cluster_idx]
        
        # compute distance between each vector and the mean vector
        cluster_mean = np.mean(cluster, axis=0)
        distances = np.linalg.norm(cluster - cluster_mean, axis=1)
        
        # sort the indices based on the distance in ascending order
        sorted_indices = np.argsort(distances)

        # aggregate results
        result += f'CLUSTER {cluster_idx} | SIZE {len(cluster)} | TOPICS {topics[cluster_idx]}:\n'
        for sentence_id in sorted_indices[:5]:
            result += f'\t{cluster_responses[sentence_id]}\n'
        result += "\t...\n" 
        for sentence_id in sorted_indices[-5:]:
            result += f'\t{cluster_responses[sentence_id]}\n'
        result += "\n"
    
    # save results to a .txt file
    with open(save_dir + 'results.txt', 'w') as f:
        f.write(result)
        
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', type=str, default='./datasets/expect_from_peers.xlsx', 
                        help='Path to excel (.xlsx) file containing survey responses.')
    
    parser.add_argument('--sent', type=str, default='true', 
                        help='Whether or not to split up the input text into sentences.')
    
    parser.add_argument('--ignore_words', nargs='+', default=['expect', 'peers'],
                        help='Words to ignore when extracting cluster topics.')
    
    parser.add_argument('--model', type=str, default='all-MiniLM-L12-v2', 
                        help='The embedding model from sentence-transformers.')
  
    parser.add_argument('--thresh', type=float, default=2.5,
                        help='The threshold to use for agglomerative clustering.')
    
    parser.add_argument('--save_dir', type=str, default='./example/', 
                        help='The directory to save results in.')
    
    args = parser.parse_args()
    
    try:
        os.mkdir(args.save_dir)
    except:
        pass
    
    sent = (args.sent == 'true')
    
    # clustering
    responses, classes, embeddings = cluster(data_path=args.data_path,
                                           model=args.model,
                                           sent=sent,
                                           thresh=args.thresh)
    
    # extract topics
    topics = extract_topics(text_data=responses, 
                            classes=classes, 
                            apply_tfidf=True, 
                            ignore_words=args.ignore_words)
    
    # save results
    result = save_results(responses=responses,
                          classes=classes,
                          embeddings=embeddings,
                          topics=topics,
                          save_dir=args.save_dir)
    
    print(result)
    
    with open(args.save_dir + 'params.txt', 'w') as f:
        f.write(f"Parameters:\n\ndata_path: {args.data_path}\nsent: {sent}\nignore_words: {args.ignore_words}\nmodel: {args.model}\nthresh: {args.thresh}\n")
        