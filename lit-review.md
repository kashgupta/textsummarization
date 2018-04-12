# CIS 530 - Computational Linguistics
#### Literature Review
### Team members:
- Ignacio Arranz
- Garvit Gupta
- Kashish Gupta
- Aditya Kashyap
- Shiva Suri
- Jialin Wang

### Implementation

We chose to implement the atomic event-based published baseline because we found that three of the papers we read (event-based, QR matrix decomposition, and supervised+semi-supervised) all had a similar method of extracting events (or terms) and choosing sentences based on how many events they covered. The event-based baseline (with the static greedy algorithm) as well as being a straightforward model, is an unsupervised baseline, and there were many unsupervised methods used in the literature. Thus, we thought it would be best to build off an unsupervised model. In addition, this model involved examining many linguistic components of the text, such as POS and entity recognition, which we thought would be useful and interesting to work with. We will consider features from the other= papers as extension ideas for future milestones.

---

### Works Citated:
1. Conroy, John M, and Dianne P. O'leary. [“Text Summarization via Hidden Markov Models.”](Https://Www.researchgate.net/Publication/224890789_Text_summarization_via_hidden_Markov_models), ResearchGate, Jan. 2001
2. Filatova, Elena, and Vasileios Hatzivassiloglou. [“Event-Based Extractive Summarization.”](www.aclweb.org/anthology/W04-1017), ACL
3. Barzilay, Regina. [“Using Lexical Chains for Text Summarization”]( www.aclweb.org/anthology/W97-0703), ACL
4. Chatterjee, Niladri, and Amol Mittal. [“Single Document Extractive Text Summarization Using Genetic Algorithms.”]( www.researchgate.net/publication/260741276_Single_document_extractive_text_summarization_using_Genetic_Algorithms), ResearchGate, Nov. 2012
5. Wong, Kam-Fai, et al. ["Extractive Summarization Using Supervised and Semi-Supervised Learning"](anthology.aclweb.org/C/C08/C08-1124.pdf), ACL


### 1. Text summarization via hidden Markov models
##### The paper presents two new methods:
1. QR Matrix Decomposition
2. Hidden Markov Models

##### QR Matrix Decomposition
First document summarized by seeking the main ideas:
- SRA’s NameTagTM [Kru95] is used to recognize named entities
- WordNet is used to associate synonyms.

Once the terms are defined, a term-sentence matrix is formed. The job of the automatic summarization system is to choose a small subset of these vectors to cover the main ideas (terms) in the document.
- The QR Matrix Decomposition iteratively chooses the vector with the largest weight. Then the weights of the remaining sentences are updated to emphasize ideas not contained in the first sentence.
- Parse the document for terms using the Brill tagger [Bri93] (in this approach, terms are equivalent to ideas)
- Create a term-sentence matrix A: if word i appears in sentence j, the value of Aij should be 1. If it doesn’t, it should be 0.
- Algorithm does as follows (looping):
    - Choose a sentence by measuring the norm (euclidean length) of its column, and choose the one with highest value.
    - Remove that sentence (column) from A
Subtract 1 from all the words (rows) from A that belong to the sentence above (this is called decomposition)
    - At each stage of the algorithm, the next sentence that we choose is the one with the largest norm.
    - Create weighting factors of sentences by their position in the document:
        - Nonzeros in column j of matrix Aw are (g*exp(-8*j/n)+t) times those in A.
    - N is the number of sentences, g and t are parameters TBD.
    - Using a training (development) set of documents, the parameter t is determined so that the function above has a tail approx. the same height as the histogram of the distribution of summary sentences.
    - Given t, we determine g and the percent of the document to capture by maximizing F1. F1 = 2r/(kh + km)
    - kh is the length of the human summary, km is the length of the machine generated summary, r is the number of sentences they share in common.

##### Hidden Markov Models
The HMM has two kinds of states, corresponding to summary sentences and non-summary sentences.
Features are:
- Position of the sentence in the document
- Position of the sentence within its paragraph
- Number of terms in the sentence -> log(number of terms+1)
- How likely the terms are, given a baseline of terms -> log(Pr(terms in i | baseline)
- How likely the terms are, given the document terms -> log(Pr(terms in i | document)

The model has 2s+1 states, with s summary states and s+1 non-summary states.
The model has 2s free parameters (the arrows). The probability of transition between summary states 2j and 2j+2 is the number of times sentence j+1 directly followed summary sentence j in the training documents, divided by the number of documents.

The model has three parts:
- p, the initial state distribution
- M, the markov transition matrix
- B, the collection of multi-variant normal distributions associated with each state.

Finally, there are two method for extracting summaries with k sentences. The first one simply chooses those sentences with the maximum posterior probability of being a summary sentence. The second uses the QR decomposition described above to remove redundancies.

---

### 2. Event-Based Extractive Summarization
Filatova and Vasileios describe the identification of concepts in the text as an important step in text summarization and propose a model of concepts based on atomic events. Atomic events are information about relevant named entities and the relationships between them--specifically, a pair of named entities connected by a verb or action-indicating noun. Each atomic event is given a score (importance) based on the relative frequency of the relation and connector. They compare their event-based algorithm with an algorithm that uses the (weighted) words that appear in the text (based on presence in titles, headers, being the first sentence, proximity to words like important or significant, etc).

Filatova and Vasileios model the extractive summarization problem as breaking the text into units, extracting the concepts covered by each unit, and choosing the most minimal units that cover all the most important concepts. They try several methods of getting the text units, a basic method that gets text units that cover many concepts (static greedy), and others which try to reduce redundancy in the selected units (adaptive greedy) or focus on (weigh more highly) units that cover the most important concepts over the less important ones (modified adaptive greedy). This study used sentences as text units, atomic events and (tf-idf weighted) words present in text as features, and ROUGE as their evaluation method. With all three methods of text unit selection, the atomic event model performed better on ROUGE on more document sets than the tf-idf weighted model, and both extensions to the model (reducing redundancy and focusing on most important concepts) performed better than the basic unit selection method, with the modified greedy adaptive method resulting in improvements over the static greedy method for both the atomic event and tf-idf weighted models, but with a much greater improvement for the atomic event model, with events-based giving a better summary on 70% of document sets for summaries of 100 words, and similar results for 50, 200, and 400 words.

---

### 3. Using Lexical Chains for Text Summarization

Introduction:
Summarization is a two-step process:
1)    Building from the source text a source representation
2)    Summary generation – forming a summary representation from the source representation built in the first step and synthesizing the output summary text
 
Summaries can be built from shallow linguistic analysis:
1)    Word distribution, most frequent words represent the most important part of the text
2)    Cue phrases. eg. In conclusion
3)    Location method – heading, sentences in the beginning or the end of the text
 
Algorithm for Chain Computing:
First computational model for lexical chains was presented in the work of Morris and Hirst
1)    They define lexical cohesion relations in terms of categories, index entries and pointers in Roget’s Thesaurus.
2)    Chains are evaluated by taking a new text word and finding a related chain for it according to relatedness criteria.
3)    DRAWBACK: They did not require the same word to appear with the same sense in its different occurrences for it to belong to a chain.
 
Constructing lexical chains follows three steps:
1)    Select a set of candidate words
2)    For each candidate word, find an appropriate chain relying on relatedness criterion among members of the chains
3)    If it is found, insert the word in the chain and update it accordingly
 
In the preprocessing step:
1)    All the words that appear as a noun entry in WordNet are chosen
2)    Relatedness of the words is determined in terms of the distance between their occurrences and the shape of the path connecting them in the WordNet Thesaurus. Three kinds of relations are defined:
- Extra strong (Between a word and its repetition)
- Strong (Between two words connected by a WordNet relation)
- Medium Strong (Link between the Synsets of the words is longer than one)
Maximum distance between related words depends on the kind of relation:
- Extra strong: No limit in distance
- Strong relations: limited to a window of seven sentences
- Medium Strong: Within three sentences back

#### Disadvantage
This algorithm implements a greedy disambiguation strategy. Disambiguation cannot be a greedy decision. In order to choose the right sense of the word, the “whole picture” of chain distribution in the text must be considered
 
#### Novelty of the algorithm 
The current algorithm differs from H&S in the following ways:
- it introduces, in addition to the relatedness criterion for membership to a chain, a non-greedy disambiguation heuristic to select the appropriate senses of the chain members
- the criterion for the selection of candidate words
- the operative definition of the text unit
- Candidate words chosen: simple nouns and noun compounds
 
Building Summaries Using Lexical Chains:
- The most prevalent discourse topic will play an important role in the summary.
- Picking concepts represented by strong lexical chains gives a better indication of central topic of a text than simply picking the most frequent words in the text.
The following parameters were found to be good predictors of the strength of the chain:
1)    Length: The number of occurrences of members in the chain
2)    Homogeneity index: 1 – the number of distinct occurrences divided by the length
`Score(Chain) = Length*Homogeneity`

---

### 4. Single Document Extractive Text Summarization using Genetic Algorithms
The paper proposes that by clustering sentences which contain similar pieces of information together, we can subsequently extract strong summaries by aggregating the most important sentence from each cluster. The authors also highlight that a common problem with k-means clustering is convergence to local minima (rather than global), and hence suggest the use of “evolutionary” (i.e. genetic) algorithms for clustering.
 
The quality of a partitional clustering algorithm is determined by the similarity measure used and the optimization criterion (fitness) function chosen to reflect the underlying definition of the quality of clusters. The authors propose to use cosine similarity measure and a fitness function they developed for the sole purpose of extractive text summarization. The paper outlines 5 decisions necessary to make in order to achieve the summary via this method:
- Represent document in mathematical (vector) form
- Define similarity metric between two documents
- Determine the criteria fitness function
- Cluster on the basis of this fitness function
- Select important sentences from each cluster to include in the summary
 
The algorithm discusses the evolutionary algorithm which involves the following pseudocode, where N is the number of sentences in the summary, CR is the cross-over rate, λ is a scalar multiplier, and k is the number of clusters.

Each chromosome is represented by , which is a vector of sentences from each cluster forming a possible cluster. ‘Discrete’ differential evolution allows chromosomes to contain integer values and hence simplifies the computations. Using a tf-idf matrix, a weight is assigned to each word in a sentence, and a vector of weights hence forms a sentences. We can compare two sentences similarities by computing the cosine similarity measure between the two vectors. Furthermore, the fitness function proposed optimizes for compactness (every sentence in a cluster is similar to one another) and separability (clusters are distinct).
 
Ultimately, we chose not to proceed initially with this method for extractive text summarization due to the genetic unreliability of genetic algorithms, which have a tendency to ‘work’ eventually after endless grid-searching without a clear explanation. Furthermore, there is too much randomness with the proposed algorithm, which would unnecessarily complicate testing and make it difficult to achieve a consistently strong summarization.


--- 
### 5. Extractive Summarization Using Supervised and Semi-Supervised Learning
1. With the notion that every sentence feature has its unique contribution, the paper investigates combined sentence features for extractive summarization.
2. Investigates the effectiveness of different sentence features with supervised learning to decide which sentences are important for summarization. 
3. The sentences are ranked, and top ones are extracted to compile the final summaries.
4. Builds on top of previous work:
    1. Position and length are useful surface features. (due to Radev et al. (2004))
    2. Centroid (due to Radev et al., 2004)
    3. Signature terms (due to Lin and Hovy, 2000)
    4. High frequency words (due to Nenkova et al., 2006).
5. Framework: 
    1. Rank~i~= RankPos~i~ + RankLength~i~
    2. Where RankPosiis the rank of sentence i according to its position in a document (i.e. the sentence no.) and RankLengthi is rank of sentence i according to its length. 
6. Sentence Features for Extractive Summarization
    1. Surface Features: Based on structure of documents or sentences, including sentence position in the document, the number of words in the sentence, and the number of quoted words in the sentence.
    2. Content Features: Integrated three well-known sentence features based on content-bearing words i.e., centroid words, signature terms, and high frequency words.
    3. Event Features: An event is comprised of an event term and associated event elements.
    4. Relevance features: Incorporated to exploit inter-sentence relationships. It is assumed that: (1) sentences related to important sentences are important; (2) sentences related to many other sentences are important. The first sentence in a document or a paragraph is important, and other sentences in a document are compared with the leading ones. 

7. Models used:
    1. Probabilistic Support Vector Machine (PSVM)
    2. Naïve Bayesian Classifier (NBC)
