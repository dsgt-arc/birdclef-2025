from functools import cache

import faiss
import numpy as np
from gensim.models import KeyedVectors


@cache
def get_index(path):
    """Get the FAISS index for the centroids."""
    centroids = np.load(path)
    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(centroids)
    return index


@cache
def get_word_vectors(path):
    """Get the word vectors from the Word2Vec model."""
    return KeyedVectors.load(path)


@cache
def get_pca(path):
    """Get the PCA model from the tokenizer."""
    return faiss.read_VectorTransform(path)
