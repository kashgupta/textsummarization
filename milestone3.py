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

event_hyponyms_file = 'event_hyponyms.txt'
activity_hyponyms_file = 'activity_hyponyms.txt'

f_events = open(event_hyponyms_file, 'r')
event_hyponyms = set([line.rstrip('\n').lower() for line in f_events])

f_activities = open(activity_hyponyms_file, 'r')
activity_hyponyms = set([line.rstrip('\n').lower() for line in f_activities])

action_nouns = event_hyponyms.union(activity_hyponyms)

with open(filename, "r") as f:
	data = f.read()

#WE are only doing the first 200 articles, so that it runs quickly
number_articles = 200
articles = data.split("\n")[:number_articles]

y_pred = []

#https://spacy.io/usage/linguistic-features

article_matrix = []
sentence_index_dict = {}
sentence_num = 0

# cnt_all = []
# # find top 10 most frequent nouns
# for article in articles:
#     cnt = Counter()
#     doc = nlp(article)
#     for tok in doc:
#         if tok.pos_ == 'NOUN':
#             cnt[str(tok).lower()] += 1
#     cnt_all.append(dict(cnt.most_common(10)))

for article in articles:
    doc = nlp(article)
    
    article_dict = {}          # ADDED

    #FIND TOP 10 NOUNS FOR THIS ARTICLE
    cnt = Counter()
    for tok in doc:
        if tok.pos_ == 'NOUN':
            #print('NOUN',type(tok))
            cnt[tok] += 1

    top10_dict = dict(cnt.most_common(10))
    top10_list = list(top10_dict.keys())

    sentences = list(doc.sents)
    # id_to_sentence = {id: sentence for (id, sentence) in zip(range(len(sentences)), sentences)}
    for sentence in sentences:
        sentence_index_dict[sentence_num] = sentence
        sentence = str(sentence)
        spacy_sentence = nlp(sentence)

        entities_list = list(spacy_sentence.ents)
        
        # entities_set = set(entities_list)

        #make list of both entities and top 10 nouns
        full_entities_list = entities_list + top10_list
        full_entities_string = [str(ent) for ent in full_entities_list]

        sentence_words = sentence.split(' ')

        full_entities_ordered_list = []

        #GETS THE ENTITIES AND TOP 10 NOUNS IN THE ORDER IN WHICH THEY APPEAR (ESSENTIAL FOR NEXT STEP)

        for entity in full_entities_list:
            if str(entity) in sentence:
                full_entities_ordered_list.append(entity)


        #RENAME AND COUNT
        sentence_entities = full_entities_ordered_list
        entities_count = len(full_entities_ordered_list)

        if entities_count >= 2:
            # for every consecutive pair of entities, we get the pair (atomic candidate) and the connector
            for i in range(entities_count-1):
                ent1 = sentence_entities[i]
                ent2 = sentence_entities[i+1]
                if type(ent1) == spacy.tokens.token.Token:
                    A1 = int(ent1.idx)
                    A2 = int(ent1.idx)+int(len(ent1))
                else:
                    A1 = int(ent1.start_char)
                    A2 = int(ent1.end_char)

                if type(ent2) == spacy.tokens.token.Token:
                    B1 = int(ent2.idx)
                    B2 = int(ent2.idx)+int(len(ent2))
                else:
                    B1 = int(ent2.start_char)
                    B2 = int(ent2.end_char)

                
                atomic_candidate = sentence[A1:B2]
                connector = sentence[A2:B1]

                # check whether connector has verbs
                spacy_connector = nlp(connector)
                connector_has_verb = False
                for token in spacy_connector:
                    if token.pos_ == 'VERB' or (token.pos_ == 'NOUN' and str(token).lower() in action_nouns):
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


