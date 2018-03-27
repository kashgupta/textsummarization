
def rouge_metrics(system_list,reference_list):
	reference_word_count = len(reference_list)
	system_word_count = len(system_list)
	rouge_recall = len(intersection(system_list,reference_list))/reference_word_count
	rouge_precision = len(intersection(system_list,reference_list))/system_word_count

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


system_text = "the cat was found under the bed"
reference_text = "the cat was under the bed"


system_list = system_text.split(" ")
reference_list = reference_text.split(" ")


system_2grams = create_ngrams(system_list,2)
reference_2grams = create_ngrams(reference_list,2)


rouge_recall, rouge_precision = rouge_metrics(system_list,reference_list)


print(system_2grams)
print("recall",rouge_recall)
print("precision",rouge_precision)

#chris said that averaging this metric for every text was fine

#chris said that averaging this metric for every text was fine




