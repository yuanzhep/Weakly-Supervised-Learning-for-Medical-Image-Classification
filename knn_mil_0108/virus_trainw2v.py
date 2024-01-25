# -*- coding: utf-8 -*-
"""
Dec 29
@yz

Created on Thu Nov 24 14:15:25 2022
@author: caixd
"""

import gensim
import tempfile


def seq2sent(filename, readlen, readnum, k):
    #convert  sequences or reads to sentences 
    #filename: the file that contain sequences (reads) of 300 bps
    #readlen: length of reads 
    #readnum: number of reads, should be less than the number of reads in the file (6000)
    #k: k-mer as a word    
    sentences=[]
    with open(filename, 'r' ) as freads:
        for i, line in enumerate(freads):
            temp=line.strip()
            oneline=[]
            nwords=readlen-k+1
            #nwords=6
            for j in range(nwords):
                oneline.append(temp[j:j+k])
            
            sentences.append(oneline)
            if i >= readnum-1:
                break
            
    return sentences

#def main():
readlen=300
k=8
readnum=6000
    
filename="SampleVirusReadsH.txt"
sentences=seq2sent(filename, readlen, readnum, k)
    
filename="SampleVirusReadsNH.txt"
temp=seq2sent(filename, readlen, readnum, k)
for sent in temp:
    sentences.append(sent)
del temp

    
#train the model
model=gensim.models.Word2Vec(sentences=sentences,min_count=1)
model.save("genome.w2vmodel")


'''
#check the trained model
vec = model.wv['ATAAT']
print(model.wv.most_similar(positive=['ATAAT'], topn=5))


for index, word in enumerate(model.wv.index_to_key):
    if index == 10:
        break
    print(f"word #{index}/{len(model.wv.index_to_key)} is {word}")


if __name__ == "__main__":
    main()
'''