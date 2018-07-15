from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

def find_nn_cos(v, Wv, k=10):
    """Find nearest neighbors of a given word, by cosine similarity.
    
    Returns two parallel lists: indices of nearest neighbors, and 
    their cosine similarities. Both lists are in descending order, 
    and inclusive: so nns[0] should be the index of the input word, 
    nns[1] should be the index of the first nearest neighbor, and so on.
    
    You may find the following numpy functions useful:
      np.linalg.norm : take the l2-norm of a vector or matrix
      np.dot : dot product or matrix multiplication
      np.argsort : get indices sorted by element value,
        so np.argsort(numbers)[-5:] will return the top five elements
    
    Args:
      v: (d-dimensional vector) word vector of interest
      Wv: (V x d matrix) word embeddings
      k: (int) number of neighbors to return
    
    Returns (nns, ds), where:
      nns: (k-dimensional vector of int), row indices of nearest neighbors, 
        which may include the given word.
      similarities: (k-dimensional vector of float), cosine similarity of each 
        neighbor in nns.
    """
    pass
    #### YOUR CODE HERE ####
    nns = []
    ds = []
    
    mid = np.dot(Wv, v)
    norm_Wv = np.linalg.norm(Wv, axis=1)
    norm_v = np.linalg.norm(v, axis=-1)
    # print (norm_Wv, norm_v)
    neighbors = np.divide(mid, np.dot(norm_Wv, norm_v))
    nns = np.argsort(neighbors)[-k:]
    for n in nns:
        ds.append(neighbors[n])
    # print(nns, ds)
    return nns, ds
    #### END(YOUR CODE) ####


def analogy(vA, vB, vC, Wv, k=5):
    """Compute a linear analogy in vector space, as described in the async.

    Find the vector(s) that best answer "A is to B as C is to ___", returning 
    the top k candidates by cosine similarity.
    
    Args:
      vA: (d-dimensional vector) vector for word A
      vB: (d-dimensional vector) vector for word B
      vC: (d-dimensional vector) vector for word C
      Wv: (V x d matrix) word embeddings
      k: (int) number of neighbors to return

    Returns (nns, ds), where:
      nns: (k-dimensional vector of int), row indices of the top candidate 
        words.
      similarities: (k-dimensional vector of float), cosine similarity of each 
        of the top candidate words.
    """
    pass
    #### YOUR CODE HERE ####
    v = vC + (vB-vA)
    return find_nn_cos(v, Wv, k)

    #### END(YOUR CODE) ####
    
def show_nns(hands, word, k=10):
    """Helper function to print neighbors of a given word."""
    word = word.lower()
    print("Nearest neighbors for '{:s}'".format(word))
    v = hands.get_vector(word)
    for i, sim in zip(*vector_math.find_nn_cos(v, hands.W, k)):
        target_word = hands.vocab.id_to_word[i]
        print("{:.03f} : '{:s}'".format(sim, target_word))
    print("")
    return
    
def show_analogy(hands, a, b, c, k=5):
    """Compute and print a vector analogy."""
    a, b, c = a.lower(), b.lower(), c.lower()
    va = hands.get_vector(a)
    vb = hands.get_vector(b)
    vc = hands.get_vector(c)
    print("'{a:s}' is to '{b:s}' as '{c:s}' is to ___".format(**locals()))
    for i, sim in zip(*vector_math.analogy(va, vb, vc, hands.W, k)):
        target_word = hands.vocab.id_to_word[i]
        print("{:.03f} : '{:s}'".format(sim, target_word))
    print("")
    return
