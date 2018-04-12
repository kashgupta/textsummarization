To run the file, these are the main arguments needed:
1) --events event_hyponyms.txt
2) --activities activity_hyponyms.txt
3) --summary_length (number of sentences, which should be set to 3)
4) --pred_file (Name of the file you want to save the predications in, say m3_baseline.txt)
5) --test_file X_data_train.txt

event_hyponyms.txt and activity_hyponyms.txt have been used to find action nouns needed for the algorithm. These were obtained by scraping https://www.powerthesaurus.org. We used the python script hyponym_scraper.py for this. Both text files and the scraping file have been submitted.

The baseline can be run with:

python3 baseline.py --events event_hyponyms.txt --activities activity_hyponyms.txt --summary_length 3 --pred_file m3_baseline.txt  --test_file X_data_train.txt

For evaluation, use the python script score.py submitted earlier (we have re submitted as well), with the following script:

python3 score.py --goldfile y_data_train_500.txt --predfile m3_baseline.txt --ngram 2

The file y_data_train_500.txt is also part of our submission. Since it takes a long time to obtain the summaries, we have limited it to the first 500 articles.

On the test set, when evaluating with bigrams we obtained the following values:
		Published Baseline		Random Baseline
Precision:	0.058 				0.072
Recall: 	0.126				0.038
FScore: 	0.079				0.049

And when evaluating with unigrams, we obtained the following:
		Published Baseline		Random Baseline
Precision:	0.253 				0.305
Recall: 	0.561				0.163
FScore: 	0.349				0.212

