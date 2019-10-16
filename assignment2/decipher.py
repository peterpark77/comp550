import nltk
import nltk.tag.hmm
from nltk.corpus import treebank
import nltk.probability
import string
from nltk import ConditionalFreqDist, FreqDist
from nltk.probability import LaplaceProbDist, RandomProbDist, MLEProbDist
import sys

cipher_folder = sys.argv[-1]
train_cipher_path = cipher_folder + '/train_cipher.txt'
train_plain_path = cipher_folder + '/train_plain.txt'
test_cipher_path = cipher_folder + '/test_cipher.txt'
test_plain_path = cipher_folder + '/test_plain.txt'

with open(train_cipher_path, 'r', encoding='utf-8', errors='ignore') as somefile:
    train_cipher = somefile.readlines()
with open(train_plain_path, 'r', encoding='utf-8', errors='ignore') as somefile:
    train_plain = somefile.readlines()

train_cipher = [x.strip('\n') for x in train_cipher]
train_plain = [x.strip('\n') for x in train_plain]

train_set = []
for x,y in zip(train_cipher, train_plain):
    item = list(zip(x,y))
    train_set.append(item)

with open(test_cipher_path, 'r', encoding='utf-8', errors='ignore') as somefile:
    test_cipher = somefile.readlines()
with open(test_plain_path, 'r', encoding='utf-8', errors='ignore') as somefile:
    test_plain = somefile.readlines()
test_cipher = [x.strip('\n') for x in test_cipher]
test_plain = [x.strip('\n') for x in test_plain]

test_set = []
for x,y in zip(test_cipher, test_plain):
    item = list(zip(x,y))
    test_set.append(item)

symbols = list(string.ascii_lowercase) + [',', '.', ' ']
states = list(string.ascii_lowercase) + [',', '.', ' ']

def bigram_dist(symb):
    pos = open('rt-polarity.pos', 'r', encoding='utf-8', errors='ignore').read()
    neg = open('rt-polarity.neg', 'r', encoding='utf-8', errors='ignore').read()
    sent = neg.split('\n') + pos.split('\n')
    sent = [list(x.lower()) for x in sent]

    final_sent = []
    for s in sent:
        final_sent.append([c for c in s if c in symb])

    bigrams = []
    for s in final_sent:
        bigrams += list(nltk.bigrams(s))

    return ConditionalFreqDist(bigrams)

if '-lm' in sys.argv and '-laplace' in sys.argv:
    cond_freq = bigram_dist(symbols)
    trainer = nltk.tag.hmm.HiddenMarkovModelTagger(states=states, symbols=symbols, transitions=cond_freq, outputs=None, priors=None)
    tagger = trainer.train(labeled_sequence=train_set, estimator=LaplaceProbDist)
elif '-laplace' in sys.argv:
    trainer = nltk.tag.hmm.HiddenMarkovModelTrainer(states=states, symbols=symbols)
    tagger = trainer.train_supervised(labelled_sequences=train_set, estimator=LaplaceProbDist)
elif '-lm' in sys.argv:
    cond_freq = bigram_dist(symbols)
    trainer = nltk.tag.hmm.HiddenMarkovModelTagger(states=states, symbols=symbols, transitions=cond_freq, outputs=None, priors=None)
    tagger = trainer.train(labeled_sequence=train_set, estimator=MLEProbDist)
else:
    trainer = nltk.tag.hmm.HiddenMarkovModelTrainer(states=states, symbols=symbols)
    tagger = trainer.train_supervised(labelled_sequences=train_set)

print('Accuracy:  {}'.format(100*tagger.evaluate(test_set)))

# Comment/uncomment the following line if you don't/do want to print out the deciphered test set
print(tagger.test(test_set, verbose=True))