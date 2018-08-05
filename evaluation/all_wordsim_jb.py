import sys
import os
import numpy as np

from read_write import read_word_vectors
from ranking import *

if __name__=='__main__':
    # potentially multiple word vec files (biased, debiased, different embeddings)
    word_vec_dir = sys.argv[1]
    #word_vec_file = sys.argv[1]
    word_sim_dir = sys.argv[2]
    
    print '================================================================================='
    print "%6s" %"Serial", "%30s" % "Embeddings", "%15s" % "RG (##/65)", "%15s" % "WS (##/353)"#, "%15s" % "Rho"
    print '================================================================================='

    word_vec_files = [f for f in os.listdir(word_vec_dir) if not f.startswith('.')] # Don't read .DS_Store!
    num_embeddings = len(word_vec_files)
    
    word_sim_files = [f for f in os.listdir(word_sim_dir) if not f.startswith('.')] # Don't read .DS_Store!
    num_benchmarks = len(word_sim_files)

    header = ["File #","Word_embedding"] + word_sim_files
    
    data = np.zeros((num_embeddings,num_benchmarks))


    for i,word_vec_file in enumerate(word_vec_files):
        # print(os.path.join(word_vec_dir,word_vec_file))
        word_vecs = read_word_vectors(os.path.join(word_vec_dir,word_vec_file))

        print "%6s" % str(i+1), "%30s" % word_vec_file,


        for j, word_sim_file in enumerate(word_sim_files):
            manual_dict, auto_dict = ({}, {})
            not_found, total_size = (0, 0)
            for line in open(os.path.join(word_sim_dir, word_sim_file),'r'):
                line = line.strip().lower()
                word1, word2, val = line.split()
                if word1 in word_vecs and word2 in word_vecs:
                    manual_dict[(word1, word2)] = float(val)
                    auto_dict[(word1, word2)] = cosine_sim(word_vecs[word1], word_vecs[word2])
                else:
                    not_found += 1
                    total_size += 1
            data[i,j] = spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict))

            print "%15.4f" % data[i,j],#, "%20s" % str(total_size),
        print("\n")

    #   print "%15s" % str(not_found),
#   print "%15.4f" % spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict))
