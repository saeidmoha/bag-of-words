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

print(len(twenty_train.target))
print(twenty_train.target[:10])
print("-" * 10)
for t in twenty_train.target[:10]:
   print(twenty_train.target_names[t])
   #print(twenty_train.target_names[t], ":\n", "\n".join(twenty_train.data[t].split("\n")[:2]), "\n------\n")
print("-" * 10)

#print(count_vect.vocabulary_.get(u'algorithm'))


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


from treeinterpreter import treeinterpreter as ti

#predicted = clf.predict(vectorizer.transform(twenty_test.data).toarray())
prediction, bias, contributions = ti.predict(clf, vectorizer.transform(twenty_test.data).toarray())

for i in range(len(docs_test)):
    print ("Instance", i)
    print ("Bias (trainset mean)", biases[i])
    print ("Feature contributions:")
    '''
    for c, feature in sorted(zip(contributions[i], 
                                 docs_test.feature_names), 
                             key=lambda x: -abs(x[0])):
        print (feature, round(c, 2))
    '''
    print ("-"*20) 

