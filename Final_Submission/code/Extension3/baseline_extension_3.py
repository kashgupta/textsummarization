from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn import svm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

import datetime

start = datetime.datetime.now()


X_train_df = pd.read_csv('Training_Data_Extension_3.csv')
X_test_full_df = pd.read_csv('Test_Data_Extension_3.csv')


y_train_df = pd.DataFrame(X_train_df['Label'],columns=["Label"])
X_train_df = X_train_df.drop(columns=['Label'])


X_test_ids = X_test_full_df['article-sentence'].tolist()
X_test_articles = [x.split("-")[0] for x in X_test_ids]


y_test_df = pd.DataFrame(X_test_full_df['Label'],columns=["Label"])
X_test_df = X_test_full_df.drop(columns=['Label','article-sentence'])


# print(X_train_df.head(10))
# print(y_train_df)

y_train_np = np.array(y_train_df).squeeze()
X_train_np = np.array(X_train_df)

y_test_np = np.array(y_test_df).squeeze()
X_test_np = np.array(X_test_df)

# clf = MLPRegressor(solver='sgd', alpha=1e-5, hidden_layer_sizes=(100, 100, 100), max_iter=1000, shuffle=True)
clf = AdaBoostClassifier(n_estimators=20, algorithm='SAMME.R')
#clf = LogisticRegression()
#clf = RandomForestRegressor(n_estimators=20, max_depth=5)
#clf = RandomForestClassifier(max_depth=4, random_state=0)
#clf = svm.SVR(kernel='linear', C=1e3, gamma=0.1)

scaler = MinMaxScaler()
X_train_np = scaler.fit_transform(X_train_np)

clf.fit(X_train_np, y_train_np)


X_test_np = scaler.transform(X_test_np)
value = clf.predict(X_test_np)
#value = clf.predict_proba(X_test_np)

#value = value[:,0]

#print('output')

value = value.tolist()

final_df = pd.DataFrame(
    {'article': X_test_articles,
    'index': X_test_ids,
     'score': value
    }).set_index('index', drop=True)

# print('final_df')
# print(final_df)

# output_df = final_df.sort_values(['article','sentence','score'], ascending=[1,0]).groupby('article').head(3)

output_df = final_df.sort_values(['article','score'], ascending=[1,0]).groupby('article').head(3)



# t = df.groupby(['borough', 'title']).sum()
# t.sort('total_loans', ascending=True)
# t = t.groupby(level=[0,1]).head(3).reset_index()
# t.sort(['borough', 'title'], ascending=(True, False))


# print(value)
# print('set',y_test_set)

# print('output_df')
# print(output_df)

selected_sentences = output_df.index.values
selected_sentences = selected_sentences.tolist()

# print('selected_sentences')
# print(selected_sentences)


y_pred = [1 if x in selected_sentences else 0 for x in X_test_ids]

y_pred_df = pd.DataFrame(
    {'y_pred': y_pred,
    })

# print('y_pred_df')
# print(y_pred_df.head(20))


f = open("entity_scores_test.txt","r")

sentences = []
article_ids = []
sentence_ids = []

article_id = 0
sentence_id = 0
summary = ""
summaries = []

for line in f:
	article_num = line.split("@@@")[0]
	text = line.split("@@@")[2].strip()

	if article_num != article_id:
		sentence_id = 0
		summaries.append(summary)
		summary = ""
		article_id = article_num
	else:
		sentence_id += 1

	new_id = str(article_num.strip())+"-"+str(sentence_id)
	if new_id in selected_sentences:
		summary += text


summaries = summaries[1:]
#print(summaries)

with open("y_pred_extension_3.csv","w") as f:
	for line in summaries:
		f.write(line)
		f.write("\n")


end = datetime.datetime.now()

time = end - start
print(time)

