
# coding: utf-8

import nltk
import random
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()

parser.add_argument("--p", type = str, default = "lemmatize", help = "type of processing")
parser.add_argument("--sw", type = int, default = 0, help = "remove stopwords" )
parser.add_argument("--t", type = float, default = 0.0001, help = "threshold for removing infrequent words")
parser.add_argument("--a", type = float, default = 1.0, help = "smoothing parameter for Naive Bayes",)
parser.add_argument("--r", type = float, default = 1.0, help = "Penalization term for Logistic regression and SVM")

args = parser.parse_args()


#define confusion matrix from scikit-learn examlple


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], ".2f"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


if args.sw == 1:
    stopwords = stopwords.words("english")
else:
    stopwords = []

# load the reviews
neg = [l.strip("\n") for l in open("./rt-polaritydata/rt-polarity.neg", "r", encoding = "ISO-8859-1")]
pos = [l.strip("\n") for l in open("./rt-polaritydata/rt-polarity.pos", "r", encoding = "ISO-8859-1")]

#define processor

def processor(string, typeofprocess):
    string = string.lower() # downcase
    tokens = nltk.tokenize.word_tokenize(string)
    if typeofprocess == "lemmatize" :
        lemmatizer = nltk.stem.WordNetLemmatizer()
        tokens = [t for t in tokens if t.isalpha()] #remove non alphabetic term
        output_tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stopwords] # remove stopwords
    elif typeofprocess == "stem" :
        stemmer = nltk.stem.PorterStemmer()
        output_tokens = [stemmer.stem(t) for t in tokens if t not in stopwords]
    else:
        print("invalid type")
    return output_tokens



y_pos = [1 for i in range(0, len(neg))]
y_neg = [-1 for i in range(0, len(pos))]
Y = y_pos
Y.extend(y_neg)
X = pos
X.extend(neg)

data = [(i,j) for i,j in zip(X,Y)]
X_data = [" ".join(processor(i[0], args.p)) for i in data]
Y_data = [i[1] for i in data]
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.33, random_state=42, shuffle = True)

Y = y_pos
Y.extend(y_neg)

#convert by count
cv = CountVectorizer(min_df = args.t)
X_train_processed = cv.fit_transform(X_train)
#load models
lr = LogisticRegression(C = args.r)
SVM = svm.SVC(kernel='linear', max_iter= 10000, tol = 1e-5, C = args.r)
NB = MultinomialNB(alpha = args.a)

#fit models
lr.fit(X_train_processed, y_train)
SVM.fit(X_train_processed, y_train)
NB.fit(X_train_processed.toarray(), y_train)

y_pred_lr = lr.predict(cv.transform(X_test))
y_pred_svm = SVM.predict(cv.transform(X_test))
y_pred_NB = NB.predict(cv.transform(X_test).toarray())
y_pred_bl = np.array([random.choice([1,-1]) for i in range(0, len(y_test))])

print("The accuracy for logistic regression is : " , sum(y_pred_lr == y_test)/len(y_test))
print("The accuracy for Linear SVM is : " , sum(y_pred_svm == y_test)/len(y_test))
print("The accuracy for Naive-Bayes is : " , sum(y_pred_NB == y_test)/len(y_test))
print("The accuracy for Baseline-model is : ", sum([i == 1 for i in y_pred_bl])/len(y_test))

class_names = ["Positive", "Negative"]
plot_confusion_matrix(y_test, y_pred_lr, classes=class_names,
                      title='Confusion matrix, Logistic Regression')
plot_confusion_matrix(y_test, y_pred_svm, classes=class_names,
                      title='Confusion matrix, Linear SVM')
plot_confusion_matrix(y_test, y_pred_NB, classes=class_names,
                      title='Confusion matrix, Naive-Bayes')

plt.show()

