#!/bin/sh
gawk '(ARGIND==1){a[$1]=1}(ARGIND==2){if($1 in a){print $0}}' words_alpha_clean.txt RANE_300d.txt | sed -n '1,50000p' | grep '[[:alpha:]]' > RANE_300d_english_50k.txt
