import re

def clean_data(line):
    line = line.split()
    line = [word.lower() for word in line]
    line = [re.sub("[,-\.!?]"," ",word) for word in line] # Remove punctuation in the text
    #line = [re.sub("[\(][a-zA-z0-9]*[\)]"," ", word) for word in line] # Remove words within brackets 
    #line = [re.sub("[0-9]*[.]*[0-9]*"," ", word) for word in line]
    return " ".join(line)

#\.-!?


filename = "../X_data_train.txt"

with open(filename,"r") as f:
	data = f.read()

lines = data.split("\n")

y_pred = []

for line in lines:
	sentences = line.split(".")
	summary = sentences[0]
	y_pred.append(summary)

y_pred = [clean_data(summary) for summary in y_pred]


with open("../baseline.txt","w") as f:
	for line in y_pred:
		f.write(line)
		f.write("\n")

