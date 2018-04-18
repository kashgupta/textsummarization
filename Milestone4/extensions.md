For this milestone, we implemented an extension to our milestone that used the same principle that sentences that covered more concepts would be more important. However, instead of whether a sentence contains atomic events, this extension considers the frequency of words that appear in the article.

The model first finds all the non-stop words (stop words being common but essentially meaningless words like "the" or "and") and gives them a weight based on their relative frequency in the article text.

Next, the model will give each sentence a weight based on the sum of the weights of each non-stop word that appears in the article. The top N sentences are selected to be the summary. We tested this extension with N = 3.

This extension is based on the model implemented in the QR decomposition/Hidden Markov paper. We felt that a word-frequency based model might show improvement compared to an atomic-event based model because:
1. It is possible that important objects/agents in the article are not named entities (although top 10 nouns being added to the named entities handles this to a certain extent, lower frequency nouns could still be important)
2. Important entities in the article will not neccessarily be part of an atomic event. For example, "At the hospital at 10:30 AM, Mary died." This is likely an important sentence that gets at the main idea of an article without extraneous information, but there is no chance it can be selected in the baseline because it does not contain any atomic events.

This extension also serves as a possible tool/feature for a future extension that turns our unsupervised model into a supervised model.

On ROUGE-2:
* Baseline: Precision: 0.126, Recall: 0.038, F-score: 0.079
* Extension: Precision = 0.051, Recall = 0.126, F-score = 0.072

On ROUGE-1:
* Baseline:  Precision: 0.253, Recall: 0.561, and F-score 0.349
* Extension: Precision = 0.236, Recall = 0.601, F-score = 0.339

Although this extension overall performed worse on the f-score and precision, with less than half the bigram precision of the baseline, the f-scores remained close and we saw near-quadrupling of the bigram recall and a smaller increase on the unigram recall. This suggests that combining the two methods, perhaps using weightings chosen by a supervised classifier, would be effective in improving the overall score. One explanation for the higher-recall, lower-precision might be that the model is biased towards longer sentwnces, since they are likely to contain more concepts. This is a problem that our baseline also shows, although it seems more pronounced in this implementation. This suggests that implementing a method of normalizing sentence weights by sentence length could be effective.

