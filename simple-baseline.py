import re
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--goldfile', type=str, required=True)
parser.add_argument('--predfile', type=str, required=True)

args = parser.parse_args()

def clean_data(line):
    line = line.split()
    line = [word.lower() for word in line]
    line = [re.sub("[,-\.!?]","",word) for word in line] # Remove punctuation in the text
    #line = [re.sub("[\(][a-zA-z0-9]*[\)]"," ", word) for word in line] # Remove words within brackets 
    #line = [re.sub("[0-9]*[.]*[0-9]*"," ", word) for word in line]
    return " ".join(line)

#\.-!?

lines = []
with open(args.goldfile,"r") as f:
    for row in f:
        line = row.split("\t")
        lines.append(line[0])


y_pred = []

for line in lines:
    sentences = line.split(".")
    summary = sentences[0]
    y_pred.append(summary)

y_pred = [clean_data(summary) for summary in y_pred]


with open(args.predfile,"w") as f:
    for line in y_pred:
        f.write(line)
        f.write("\n")
