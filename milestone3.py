import spacy
import argparse
import re
import numpy as np

parser = argparse.ArgumentParser()
nlp = spacy.load('en')

parser.add_argument('--goldfile', type=str, required=True)
parser.add_argument('--predfile', type=str, required=True)

args = parser.parse_args()

def clean_data(line):
    line = line.split()
    line = [word.lower() for word in line]
    line = [re.sub("[,-\.!?]"," ",word) for word in line] # Remove punctuation in the text
    #line = [re.sub("[\(][a-zA-z0-9]*[\)]"," ", word) for word in line] # Remove words within brackets
    #line = [re.sub("[0-9]*[.]*[0-9]*"," ", word) for word in line]
    return " ".join(line)

#using this for now because it's smaller than training set
filename = "sumdata/bothdev.txt"

with open(filename,"r") as f:
	data = f.read()

articles = data.split("\n")

y_pred = []

#https://spacy.io/usage/linguistic-features

for article in articles:
    doc = nlp(article)
    sentences = list(doc.sents)
    entities = {}
    for ent in doc.ents:
        ent = str(ent)
        ent_list = ent.split(' ')
        if len(ent_list) == 1:
            entities[ent_list[0]] = ''
        else:
            entities[ent_list[0]] = ' '.join(' ' + word for word in ent_list[1:len(ent_list)])

    sent_concept_matrix = np.array();
    id_to_sentence = {id: sentence for (id, sentence) in zip(range(len(sentences)), sentences)}
    for sentence in sentences:
        num_entities = 0
        for word in sentence.split(' '):
            if word in entities:
                if entities[word] in sentence:
                    num_entities += 1
        if num_entities >= 2:
            #this is where im stopping for the night
            #find all pairs of named entities
            #find the connector between the pair, keep the part that is a verb
            #output the named entity pair and connector that is in this sentence

    summary = ''
    y_pred.append(summary)

y_pred = [clean_data(summary) for summary in y_pred]


with open("../baseline.txt","w") as f:
	for line in y_pred:
		f.write(line)
		f.write("\n")

