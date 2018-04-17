import spacy
import argparse
import numpy as np
from collections import Counter
from collections import defaultdict
import datetime
import operator

parser = argparse.ArgumentParser()

#To run spacy, in command line: pip install spacy
#python -m spacy download en

nlp = spacy.load('en')

parser.add_argument('-d', type=str, required=True, dest = 'document')
parser.add_argument('-o', type=str, required=True, dest = 'output')
parser.add_argument('-l', type=int, required=True, dest = 'summary_length')

args = parser.parse_args()

def get_sentences(document):

	# Input:	Path of the original document
	# Returns:	List of sentences, sgmented using spaCy

	doc = open(document, 'r')
	sentences = []

	for paragraph in doc:
		tokens = nlp(paragraph)
		for sentence in tokens.sents:
			sentences.append(str(sentence))

	return sentences

def word_weights(sentences):

	# Input:	List of sentences in the document
	# Returns:	A dictionary of the normalized weight of each word in the vocabulary
	# 			Weights for stop words are set to zero

	dictionary = defaultdict(int)
	freq_dict = defaultdict(int)

	for sentence in sentences:
		for word in sentence.split():
			if(nlp.vocab[word].is_stop == False):
				dictionary[word] += 1

	summed = 0

	for k,v in dictionary.items():
		summed += v

	for k, v in dictionary.items():
		freq_dict[k] = v/summed

	return freq_dict

def sentence_weights(sentences, word_weights, n):

	# Inputs:	List of sentences in the document
	# 			The dictionary of the normalized weight for each of the words in the vocabulary
	# 			Number of sentences required in the summary
	# Returns:	The top 'n' words by their importance due to frequency of non-stop words
	
	importance = defaultdict(int)

	for sentence in sentences:
		weight = 0
		for word in sentence.split():
			weight += word_weights[word]
		importance[sentence] = weight

	sorted_x = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)[:n]

	return sorted_x

def summarize(document, n):

	# Inputs:	Path of the original document
	# 			Number of sentences required in the summary
	# Returns:	The summary of the document

	sentences = get_sentences(document)
	word_weights_dict = word_weights(sentences)
	summary_sentences = sentence_weights(sentences, word_weights_dict, n)

	summary = ""

	for sentence, importance in summary_sentences:
		summary = summary + sentence + " "

	return summary.replace("\n", "").replace("  ", " ")

# Main

'''
Inputs:			Document which needs to be summarized
				Number of sentences expected in the summary
				Output file to append the summary to

Outputs:		Appends the summary to output file

Example usage:	python3 extension.py -d original_document.txt -l 3 -o computed_summaries.txt
'''

predicted_summary = summarize(args.document, args.summary_length)

with open(args.output,"a") as f:
	for line in predicted_summary:
		f.write(line)
	f.write("\n")