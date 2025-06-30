from functools import cache

import faiss
import numpy as np
from gensim.models import KeyedVectors


@cache
def get_index(path):
    """Get the FAISS index for the centroids."""
    centroids = np.load(str(path))
    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(centroids)
    return index


@cache
def get_word_vectors(path):
    """Get the word vectors from the Word2Vec model."""
    return KeyedVectors.load(str(path))


@cache
def get_pca(path):
    """Get the PCA model from the tokenizer."""
    return faiss.read_VectorTransform(str(path))


def tokenize(S, index, pca=None):
    """Tokenize the input string using the FAISS index and optional PCA.
    S is the original 2d array of features (e.g., MFCC or melspectrogram).
    """

    X = S.astype(np.float32)
    if pca is not None:
        X = pca.apply(X)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    _, indices = index.search(X, 1)
    return indices.flatten().tolist()
