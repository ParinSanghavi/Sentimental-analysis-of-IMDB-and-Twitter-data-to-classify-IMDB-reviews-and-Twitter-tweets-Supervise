import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random

from sklearn import naive_bayes
from sklearn import linear_model
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec

stopwords = nltk.download("stopwords")          # Download the stop words from nltk

# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])

def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)

    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)        
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)

    
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)
    
    return train_pos, train_neg, test_pos, test_neg

def getDict(lol):  
    dct = dict()
    
    for l in lol:
        a = set(l)
        for s in a:
            if dct.has_key(s):
                dct[s]=dct[s]+1
            else:
                dct[s]=1      
    return dct

def getWordSet(d):
    a=set()
    for l in d:
        for w in l:
            a.add(w)
    return a

def getVector(inputted,features):
    lol=[]
    for l in inputted:
        nl=[]
        for f in features:
            if f in l:
                nl.append(1)
            else:
                nl.append(0)
        lol.append(nl)
    return lol

def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):

    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    positive=train_pos
    negative=train_neg

    pos_dict = getDict(positive)
    total_positive_word = len(pos_dict.keys())

    neg_dict = getDict(negative)
    total_negative_word = len(neg_dict.keys())

    featureList = []

    wordList = train_pos+train_neg
    words = getWordSet(wordList)

    removedStopwords = []
    for word in words:
        if word not in stopwords:
            removedStopwords.append(word)

    cond2 = []

    for word in removedStopwords:
        if word in pos_dict.keys():
            if pos_dict[word]*100/total_positive_word >= 1:
                cond2.append(word)
        elif word in neg_dict.keys():
            if neg_dict[word]*100/total_negative_word >= 1:
                cond2.append(word)

    cond3 = []

    for word in cond2:
        if word in pos_dict.keys() and word in neg_dict.keys():
            if 2*pos_dict[word] >= neg_dict[word] or 2*neg_dict[word] >= pos_dict[word]:
                cond3.append(word)
        elif word in pos_dict.keys() or word in neg_dict.keys():
            cond3.append(word)

    feature = cond3

    train_pos_vec = getVector(train_pos,feature)
    train_neg_vec = getVector(train_neg,feature)
    test_pos_vec = getVector(test_pos,feature)
    test_neg_vec = getVector(test_neg,feature)


    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE

    i=0
    labeled_train_pos = []
    for l in train_pos: 
        labeled_train_pos.append(LabeledSentence(words=l, tags=["train_pos_"+str(i)]))
        i=i+1

    i=0
    labeled_train_neg = []
    for l in train_neg: 
        labeled_train_neg.append(LabeledSentence(words=l, tags=["train_neg_"+str(i)]))
        i=i+1

    i=0
    labeled_test_pos = []
    for l in test_pos: 
        labeled_test_pos.append(LabeledSentence(words=l, tags=["test_pos_"+str(i)]))
        i=i+1

    i=0
    labeled_test_neg = []
    for l in test_neg: 
        labeled_test_neg.append(LabeledSentence(words=l, tags=["test_neg_"+str(i)]))
        i=i+1

    loop=i


    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE

    train_pos_vec = [] 
    train_neg_vec = []
    test_pos_vec = []
    test_neg_vec = []

    for x in range(0, loop):
        s = "train_pos_"+str(x)
        train_pos_vec.append(model.docvecs[s])

    for x in range(0, loop):
        s = "train_neg_"+str(x)
        train_neg_vec.append(model.docvecs[s])

    for x in range(0, loop):
        s = "test_pos_"+str(x)
        test_pos_vec.append(model.docvecs[s])

    for x in range(0, loop):
        s = "test_neg_"+str(x)
        test_neg_vec.append(model.docvecs[s])

    # Return the four feature vectors

    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE

    nb_model = BernoulliNB()
    nb_model = nb_model.fit(train_pos_vec+train_neg_vec,Y)
    BernoulliNB(alpha=1.0, binarize=None, class_prior=None, fit_prior=True)

    lr_model = LogisticRegression()
    lr_model = lr_model.fit(train_pos_vec+train_neg_vec,Y)
    LogisticRegression()
    
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE

    nb_model = GaussianNB()
    nb_model = nb_model.fit(train_pos_vec+train_neg_vec, Y)
    

    lr_model = LogisticRegression()
    lr_model = lr_model.fit(train_pos_vec+train_neg_vec,Y)
    

    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=True):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    
    pred = model.predict(test_pos_vec)

    tp = 0
    fn = 0
    for s in pred:
        if s == "pos":
            tp=tp+1
        else:
            fn=fn+1

    pred = model.predict(test_neg_vec)
    tn = 0
    fp = 0
    for s in pred:
        if s == "neg":
            tn=tn+1
        else:
            fp=fp+1

    accuracy = float((tn+tp))/float(tn+tp+fn+fp)

    
    
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)

if __name__ == "__main__":
    main()