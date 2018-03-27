
filename = "training_texts.txt"

with open(filename,"r") as f:
	data = f.read()

lines = data.split("\n")

y_pred = []

for line in lines:
	sentences = line.split(".")
	summary = sentences[0]
	y_pred.append(summary)


with open("baseline.txt","w") as f:
	for line in y_pred:
		f.write(line)
		f.write("\n")