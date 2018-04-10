# code to calculate Rouge precision and recall for various texts, by taking the two text files 
# that have the gold summaries and the predicted ones.

# Inputs:

# goldfile) File containing only gold summaries
# predfile) File containing predicted summaries
# ngram) n-gram model to use (1, 2, 3 ...) (Should be less than the total number of words in any paragraph)

# Output

# Baseline_n_gram.txt containing tab delimited Precision and Recall values for each pair.

import argparse
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument('--goldfile', type=str, required=True)
parser.add_argument('--predfile', type=str, required=True)
parser.add_argument('--ngram', type=int, required=True)

args = parser.parse_args()

def rouge_metrics(system_list,reference_list):
    reference_word_count = len(reference_list)
    system_word_count = len(system_list)
    if (system_word_count == 0) or (reference_word_count == 0):
        rouge_recall = 0
        rouge_precision = 0
    else:
        rouge_recall = (len(intersection(system_list,reference_list))*1.0)/reference_word_count
        rouge_precision = (len(intersection(system_list,reference_list))*1.0)/system_word_count
    return rouge_recall, rouge_precision

def intersection(system_lst, ref_lst):
    intersection_lst = [value for value in system_lst if value in ref_lst]
    return intersection_lst

def create_ngrams(text_list,n=2):
    iterations = len(text_list)-n
    ngrams = []
    gram = []
    for i in range(iterations+1):
        gram = text_list[i:n+i]
        ngrams.append(gram)

    return ngrams

with open(args.goldfile, "r") as f:
    original = f.read()

with open(args.predfile, "r") as f:
    new = f.read()

rrecall = []
rprecision = []

with open("baseline_" + str(args.ngram) + "_gram.txt", "w") as f:
    for row_original, row_new in zip(original.split("\n"), new.split("\n")):
        system_list = row_new.split(" ")
        reference_list = row_original.split(" ")

        #print ("System List:", system_list)
        #print ("Reference List:", reference_list)

        system_2grams = create_ngrams(system_list, args.ngram)
        reference_2grams = create_ngrams(reference_list,args.ngram)

        rouge_2_recall, rouge_2_precision = rouge_metrics(system_2grams,reference_2grams)

        rrecall.append(rouge_2_recall)
        rprecision.append(rouge_2_precision)

        line = str(rouge_2_recall) + "\t" + str(rouge_2_precision) + "\n"
        f.write(line)

rrecall = np.mean(rrecall)
rprecision = np.mean(rprecision)
f_score = (2*rprecision*rrecall)/(rprecision + rrecall)

print ("Recall:", rrecall)
print ("Precision:", rprecision)
print ("FScore:", f_score)
