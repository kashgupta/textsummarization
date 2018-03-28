For the evaluation script, we chose Rouge-2 (Rouge-N with N=2, using bigrams). Rouge stands for "Recall-Oriented Understudy for Gisting Evaluation".

In its most basic form, Rouge takes two summaries (a system generated summary and a reference summary), and computes the following:

Recall = (number of overlapping words)/(total words in reference summary)
Precision = (number of overlapping words)/(total words in reference summary)

For Rouge-N, the formulas above can be adapted to be:
Recall = (number of overlapping n-grams)/(total n-grams in reference summary)
Precision = (number of overlapping n-grams)/(total n-grams in reference summary)

The metrics for precision and recall can be obtained by running calculator.py.

ROUGE was first introduced for text summarization in ACL in 2003.
See paper by Chin-Yew Lin (http://www.aclweb.org/anthology/W04-1013). There it is stated that ROUGE-2, ROUGE-L and ROUGE-W worked well in single document summarization tasks, which validates our choice of ROUGE-2 as the selected evaluation method.

High scores are better. Ideally, one would want to obtain precision, recall and f-score of 1.00, which is the optimal value. However, it is interesting that a perfectly good summary may not obtain a f-score of 1.00, so it is something to take into consideration when evaluating more enhanced methods of generating the summaries.