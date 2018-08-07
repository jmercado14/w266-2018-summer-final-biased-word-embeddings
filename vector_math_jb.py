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

    # calculate cosine similarity of v with all other words in vocab
    def cos_sim(c):
        return np.dot(v,c) / (np.linalg.norm(v) * np.linalg.norm(c))
        
    sims = np.apply_along_axis(cos_sim,1,Wv)
    nns = np.argsort(sims)[-k:]
    similarities = np.sort(sims)[-k:]
    
    return(nns,similarities)


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
    v = vC - (vA - vB)
    return find_nn_cos(v, Wv, k)


def show_nns(e, word, k=10):
    """Helper function to print neighbors of a given word."""
    word = word.lower() # leave in just in case
    print("Nearest neighbors for '{:s}'".format(word))
    v = e.v(word)
    for i, sim in zip(*find_nn_cos(v, e.vecs, k)):
        target_word = e.words[i]
        print("{:.03f} : '{:s}'".format(sim, target_word))
    print("")
    
def show_analogy(e, a, b, c, k=5):
    """Compute and print a vector analogy."""
    a, b, c = a.lower(), b.lower(), c.lower()
    
    va = e.v(a)
    vb = e.v(b)
    vc = e.v(c)
   
    #print("'{a:s}' is to '{b:s}' as '{c:s}' is to ___".format(**locals()))
    candidates = analogy(va, vb, vc, e.vecs, k)
    targets = []
    for i, sim in zip(*candidates):
        target_word = e.words[i]

        # don't return the same word if it's not supposed to be the same
        if target_word == c:
            if a == b:
                targets.append(target_word)
        else:
            targets.append(target_word)
    #print("{:.03f} : '{:s}'".format(sim, target_word))

    #print("")
    targets.reverse()
    return targets
