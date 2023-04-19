# Assessing Student Feedback from Large Class Surveys via Topic Modeling

Notebook [![Open In Colab][colab-badge]][notebook]

[notebook]: <https://colab.research.google.com/drive/1mTNwulr2rKcdm2j9kCujaGN0tEdp_9ua?usp=sharing>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>

---

## About

This is a topic modeling tool designed to help professors at the University of Toronto assess student feedback from surveys in large classes they are instructing. The Colab notebooks linked above provide an interface for easy use of the tool.

By providing surveys to students in their class, professors can get feedback on various things. However, for large classes, it is difficult and time-consuming to assess this feedback simply by reading the survey answers, as there are many submissions. Using this topic modeling tool, professors can painlessly assess the feedback from surveys in classes with 1000+ students in a few seconds.

The topics that the algorithm outputs will highlight recurring ideas discovered in the survey answers in an interpretable manner. 

## How It Works

The algorithm can be broken into 4 main steps.

1. **Text preprocessing:** Given a dataset containing survey answers, the algorithm first cleans and preprocesses the text. The text is broken up into sentences.

2. **Embedding:** Using a pre-trained transformer model from [Sentence Transformers](https://www.sbert.net/docs/pretrained_models.html), these sentences are embedded into a higher dimensional semantic space and converted into vectors. 

3. **Clustering:** These lower dimensional embeddings are then clustered using [Agglomerative Clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering). 

4. **Topic extraction:** Topics are extracted for each cluster by computing the within cluster unigram frequencies, applying the class TF-IDF procedure, then selecting the most frequent words. 

## Acknowledgements

This project done in collaboration with Professor Camelia Karimianpour, Professor Vardan Papyan, and Daniel Tovbis.
