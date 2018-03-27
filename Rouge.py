
def rouge_metrics(lst1, lst2):
	rouge_recall = len(intersection(system_list,reference_list))/reference_word_count
	rouge_precision = len(intersection(system_list,reference_list))/system_word_count



def intersection(system_lst, ref_lst):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

system_text = "the cat was found under the bed"
reference_text = "the cat was under the bed"


system_list = system_text.split(" ")
reference_list = reference_text.split(" ")

system_word_count = len(system_list)
reference_word_count = len(reference_list)


rouge_recall = len(intersection(system_list,reference_list))/reference_word_count
rouge_precision = len(intersection(system_list,reference_list))/system_word_count


print("recall",rouge_recall)
print("precision",rouge_precision)

#chris said that averaging this metric for every text was fine




