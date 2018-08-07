# debiasing
from __future__ import print_function, division
import we, json, argparse
import numpy as np
import sys
from sklearn.decomposition import PCA

if sys.version_info[0] < 3:
    import io
    open = io.open

def debias(E, gender_specific_words, definitional, equalize):
    m_vd = []
    for a, b in definitional:
        center = (E.v(a) + E.v(b))/2
        m_vd.append(E.v(a) - center)
        m_vd.append(E.v(b) - center)
    m_vd = np.array(m_vd)
    pca = PCA(n_components = 10)
    pca.fit(m_vd)
    gender_direction = pca.components_[0]
    specific_set = set(gender_specific_words)
    for i, w in enumerate(E.words):
        if w not in specific_set:
            E.vecs[i] = we.drop(E.vecs[i], gender_direction)
    E.normalize()
    candidates = {x for e1, e2 in equalize for x in [(e1.lower(), e2.lower()),
                                                     (e1.title(), e2.title()),
                                                     (e1.upper(), e2.upper())]}
#     print(candidates)
    for (a, b) in candidates:
        if (a in E.index and b in E.index):
            y = we.drop((E.v(a) + E.v(b)) / 2, gender_direction)
            z = np.sqrt(1 - np.linalg.norm(y)**2)
            if (E.v(a) - E.v(b)).dot(gender_direction) < 0:
                z = -z
            E.vecs[E.index[a]] = z * gender_direction + y
            E.vecs[E.index[b]] = -z * gender_direction + y
    E.normalize()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("embedding_filename", help="The name of the embedding")
    parser.add_argument("definitional_filename", help="JSON of definitional pairs")
    parser.add_argument("gendered_words_filename", help="File containing words not to neutralize (one per line)")
    parser.add_argument("equalize_filename", help="???.bin")
    parser.add_argument("debiased_filename", help="???.bin")

    args = parser.parse_args()
    print(args)

    with open(args.definitional_filename, "r") as f:
        defs = json.load(f)

    with open(args.equalize_filename, "r") as f:
        equalize_pairs = json.load(f)

    with open(args.gendered_words_filename, "r") as f:
        gender_specific_words = json.load(f)

    E = we.WordEmbedding(args.embedding_filename)

    print("Debiasing...")
    debias(E, gender_specific_words, defs, equalize_pairs)

    print("Saving to file...")
    if args.embedding_filename[-4:] == args.debiased_filename[-4:] == ".bin":
        E.save_w2v(args.debiased_filename)
    else:
        E.save(args.debiased_filename)

    print("\n\nDone!\n")
