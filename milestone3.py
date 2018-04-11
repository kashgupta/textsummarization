import spacy
import argparse
import re
import numpy as np
import time
from collections import Counter
import datetime

start_time = datetime.datetime.now()

#parser = argparse.ArgumentParser()

#To run spacy, in command line: pip install spacy
#python -m spacy download en

nlp = spacy.load('en')

summary_length = 3


#parser.add_argument('--goldfile', type=str, required=True)
#parser.add_argument('--predfile', type=str, required=True)

#args = parser.parse_args()

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

#WE are only doing the first 200 articles, so that it runs quickly
number_articles = 200
articles = data.split("\n")[:number_articles]

y_pred = []

#https://spacy.io/usage/linguistic-features

article_matrix = []
sentence_index_dict = {}
sentence_num = 0

cnt = Counter()
# find top 10 most frequent nouns
for article in articles:
    doc = nlp(article)
    for tok in doc:
        if tok.pos_ == 'NOUN':
            cnt[str(tok).lower()] += 1
print(cnt.most_common(10))

top10nouns = dict(cnt.most_common(10))
print(top10nouns)

for article in articles:
    doc = nlp(article)
    sentences = list(doc.sents)
    article_dict = {}          # ADDED
    # id_to_sentence = {id: sentence for (id, sentence) in zip(range(len(sentences)), sentences)}
    for sentence in sentences:
        sentence_index_dict[sentence_num] = sentence
        sentence = str(sentence)
        spacy_sentence = nlp(sentence)
        entities_list = list(spacy_sentence.ents)
        entities_strings = [str(ent) for ent in entities_list]
        # entities_set = set(entities_list)

        sentence_words = sentence.split(' ')
        # scan for top 10 nouns
        top10_list = []

        for i in range(len(spacy_sentence)):
            word_str = str(spacy_sentence[i]).lower()
            if word_str in top10nouns.keys() and word_str not in entities_strings:
                span_noun = spacy_sentence[i:i+1]
                top10_list.append(span_noun)

        sentence_entities = tuple(list(entities_list + top10_list))
        entities_count = len(sentence_entities)

        if entities_count >= 2:
            # for every consecutive pair of entities, we get the pair (atomic candidate) and the connector
            for i in range(entities_count-1):
                ent1 = sentence_entities[i]
                ent2 = sentence_entities[i+1]
                A1 = int(ent1.start_char)
                A2 = int(ent1.end_char)
                B1 = int(ent2.start_char)
                B2 = int(ent2.end_char)
                atomic_candidate = sentence[A1:B2]
                connector = sentence[A2:B1]

                # check whether connector has verbs
                spacy_connector = nlp(connector)
                connector_has_verb = False
                for token in spacy_connector:
                    if token.pos_ == 'VERB':
                        connector_has_verb = True
                        break

                if connector_has_verb:
                    if sentence_num not in article_dict.keys():
                        article_dict[sentence_num] = 0

                    article_dict[sentence_num] += 1 # ADDED
#                    print('atomic_candidate',atomic_candidate)
#                    print('connector',connector)
        sentence_num += 1
    article_matrix.append(article_dict) # ADDED
#                time.sleep(1)
print(article_matrix)


for sentence_dic in article_matrix:
    #first = max(sentence_dic.iteritems(), key=operator.itemgetter(1))[0]
    sorted_dic = sorted(sentence_dic, key=lambda k: sentence_dic[k])
    
    keys = sorted_dic[:summary_length]

    temp_summary = ""
    for key in keys:
        temp_summary += str(sentence_index_dict[key])

    y_pred.append(temp_summary)


with open("m3_baseline.txt","w") as f:
	for line in y_pred:
		f.write(line)
		f.write("\n")


end_time = datetime.datetime.now()
total_time = end_time - start_time

print('total running time for '+str(number_articles)+" articles is "+str(total_time))


