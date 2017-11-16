# http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# same as previous program but using "TfidfVectorizer" instead of "CountVectorizer"
# the comments are not completely correct anymore.

from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
print(len(twenty_train.data))
print(twenty_train.target_names)
# Let’s print the first lines of the first loaded file
print("\n".join(twenty_train.data[0].split("\n")[:3]))
print(twenty_train.target_names[twenty_train.target[0]])
'''
Supervised learning algorithms will require a category label for each document in the training set. In this case the category is the name of the newsgroup which also happens to be the name of the folder holding the individual documents.

For speed and space efficiency reasons scikit-learn loads the target attribute as an array of integers that corresponds to the index of the category name in the target_names list. 
'''
print(len(twenty_train.target))
print(twenty_train.target[:10])
print("-" * 10)
for t in twenty_train.target[:10]:
   print(twenty_train.target_names[t])
   #print(twenty_train.target_names[t], ":\n", "\n".join(twenty_train.data[t].split("\n")[:2]), "\n------\n")
print("-" * 10)
'''
You can notice that the samples have been shuffled randomly (with a fixed RNG seed): this is useful if you select only the first samples to quickly train a model and get a first idea of the results before re-training on the complete dataset later.


Bags of words:

        assign a fixed integer id to each word occurring in any document of the training set (for instance by building a dictionary from words to integer indices).
        for each document #i, count the number of occurrences of each word w and store it in X[i, j] as the value of feature #j where j is the index of word w in the dictionary

The bags of words representation implies that n_features is the number of distinct words in the corpus.
This number is typically larger than 100,000.
If n_samples == 10000, storing X as a numpy array of type float32 would require 10000 x 100000 x 4 bytes = 4GB in RAM

Fortunately, most values in X will be zeros since for a given document less than a couple thousands of distinct words will be used. For this reason we say that bags of words are typically high-dimensional sparse datasets. We can save a lot of memory by only storing the non-zero parts of the feature vectors in memory.

scipy.sparse matrices are data structures that do exactly this, and scikit-learn has built-in support for these structures.

Text preprocessing, tokenizing and filtering of stopwords are included in a high level component that is able to build a dictionary of features and transform documents to feature vectors:
'''
'''
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts.shape)
print("X_train_counts[0, 32493] occurence = %d" %(X_train_counts[0, 32493]))
'''
'''
CountVectorizer supports counts of N-grams of words or consecutive characters. Once fitted, the vectorizer has built a dictionary of feature indices:
'''
#print(count_vect.vocabulary_.get(u'algorithm'))
# The index value of a word in the vocabulary is linked to its frequency in the whole training corpus.

'''
From occurrences to frequencies

Occurrence count is a good start but there is an issue: longer documents will have higher average count values than shorter documents, even though they might talk about the same topics.

To avoid these potential discrepancies it suffices to divide the number of occurrences of each word in a document by the total number of words in the document: these new features are called tf for Term Frequencies.

Another refinement on top of tf is to downscale weights for words that occur in many documents in the corpus and are therefore less informative than those that occur only in a smaller portion of the corpus.

This downscaling is called tf–idf for “Term Frequency times Inverse Document Frequency”.

Both tf and tf–idf can be computed as follows:
'''
'''
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print(X_train_tf.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)
'''
####


from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')

X_train_tfidf = vectorizer.fit_transform(twenty_train.data).toarray()
print(len(twenty_train.data))
print(twenty_train.data[0])
print(X_train_tfidf.shape)
print("count at [0,3540] is = ", X_train_tfidf[0,3540])
feature_name = vectorizer.get_feature_names()
print(len(feature_name))
print(feature_name[3540:3547])
print("index=%d  word=%s" %(feature_name.index(u'algorithm'), feature_name[feature_name.index(u'algorithm')]))

'''
In the above example-code, we firstly use the fit(..) method to fit our estimator to the data and secondly the transform(..) method to transform our count-matrix to a tf-idf representation. These two steps can be combined to achieve the same end result faster by skipping redundant processing. This is done through using the fit_transform(..) 
'''
# Now that we have our features, we can train a classifier to try to predict the category of a post. 
#from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_jobs=4, n_estimators=150) #, oob_score=True
clf.fit(X_train_tfidf, twenty_train.target)
# predict
docs_new = ['God is love', 'OpenGL on the GPU is fast']

#X_new_tfidf = vectorizer.transform(docs_new)
X_new_tfidf = vectorizer.transform(docs_new).toarray()

predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
   print('%r => %s' % (doc, twenty_train.target_names[category]))
'''
In order to make the vectorizer => transformer => classifier easier to work with, scikit-learn provides a Pipeline class that behaves like a compound classifier:
'''



from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')),('tfidf', TfidfTransformer()),('clf', RandomForestClassifier(n_jobs=4, n_estimators=50)), ])

# We can now train the model with a single command:
text_clf.fit(twenty_train.data, twenty_train.target)  

# Evaluation of the performance on the test set
import numpy as np
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target) )





