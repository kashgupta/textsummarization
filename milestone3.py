import spacy
import argparse
import re
import numpy as np
import time

parser = argparse.ArgumentParser()

#To run spacy, in command line: pip install spacy
#python -m spacy download en
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
filename = "../X_data_train_5K.txt"

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

    #sent_concept_matrix = np.array(entities);
    id_to_sentence = {id: sentence for (id, sentence) in zip(range(len(sentences)), sentences)}
    for sentence in sentences:
        #process the sentence with spacy to extract entities without for loop
        sentence = str(sentence)
        spacy_sentence = nlp(sentence)
        #print(spacy_sentence)
        sentence_entities = spacy_sentence.ents
        entities_count = len(sentence_entities)

        if entities_count >= 2:
            #for every consecutive pair of entities, we get the pair (atomic candidate) and the connector
            for i in range(entities_count-1):
                ent1 = sentence_entities[i]
                ent2 = sentence_entities[i+1]
                A1 = int(ent1.start_char)
                A2 = int(ent1.end_char)
                B1 = int(ent2.start_char)
                B2 = int(ent2.end_char)
                atomic_candidate = sentence[A1:B2]
                connector = sentence[A2:B1]

                #check whether connector has verbs
                spacy_connector = nlp(connector)
                connector_has_verb = False
                for token in spacy_connector:
                    print(token.pos_)
                    if token.pos_ == 'VERB':
                        print('it is a verb!')
                        connector_has_verb = True
                        break

                if connector_has_verb:
                    pass
                    #print('atomic_candidate',atomic_candidate)
                    #print('connector',connector)
                else:
                    print('discarded candidate',atomic_candidate)

                time.sleep(1)



            #find the connector between the pair, keep the part that is a verb
            #output the named entity pair and connector that is in this sentence

    summary = ''
    y_pred.append(summary)

y_pred = [clean_data(summary) for summary in y_pred]


with open("../baseline.txt","w") as f:
	for line in y_pred:
		f.write(line)
		f.write("\n")

