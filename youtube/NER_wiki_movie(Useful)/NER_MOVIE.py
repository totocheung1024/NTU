import sklearn_crfsuite
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from highlight_text import HighlightText, ax_text, fig_text
import random

#nltk.download('punkt')

#nltk.download('averaged_perceptron_tagger')
"""nltk stands for e Natural Language Toolkit
Punkt is an pre-train unsupervised ml tokenizer
Average perceptron tagger stands for tagging the type of word,
like vocab, noun...
"""
df = pd.read_csv('wiki_movie_plots_deduped.csv')


#taking 1k samples as test & 10k samples for training as just a demo
df_train = df[:10000]
df_test  = df[-1000:]
#Combining all plots into one single paragraph for both train & test
training = ' . '.join(df_train['Plot'])
test = ' . '.join(df_test['Plot'])

#Perform Sentence segmentation, Word tokenization & POS Tagging
#Sentence Segmentation: Separating sentences from a given chunk of text.
#Tokenization: Breaking a sentence into smaller parts (words or even smaller tokens). To know about different tokenization algorithms, do check out here
#POS Tagging: We can divide all words in a sentence into some pre-defined categories depending upon their job in the sentence used. These categories are called Part Of Speech aka POS. Verb, Noun, Adjective, etc. are some common POS tags. How is POS Tagging done? do check out here


def pos_tags(document):
        sentences = nltk.sent_tokenize(document) 
        sentences = [nltk.word_tokenize(sent) for sent in sentences]
        sentences = [nltk.pos_tag(sent) for sent in sentences]
        return sentences
training = pos_tags(training)
test = pos_tags(test)

grammar = "NP: {<DT>?<JJ>*<NN>}"

chunk_parser = nltk.RegexpParser(grammar)

""""(S
  (NP The/DT quick/JJ brown/JJ fox/NN)
  jumps/VBZ
  over/IN
  (NP the/DT lazy/JJ dog/NN))"""
training = [chunk_parser.parse(x) for x in training]
test = [chunk_parser.parse(x) for x in test]

training = [nltk.chunk.tree2conlltags(x) for x in training]
test = [nltk.chunk.tree2conlltags(x) for x in test]

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
    return features
""""
Basic Word Features: The function first extracts basic features from the current word sent[i]:

The lowercase form of the word.
The last three characters of the word (useful for recognizing suffixes).
The last two characters of the word.
Boolean features indicating if the whole word is uppercase, if the word is a title (capitalized), and if the word consists of digits.
The part-of-speech (POS) tag of the word.
The first two characters of the POS tag (which can sometimes be useful by themselves).
Contextual Features: The function also looks at the previous and next words to get contextual information:

If the current word is not the first in the sentence (i > 0), it adds features for the previous word (-1:) with similar information as mentioned above.
If the current word is the first in the sentence (i == 0), it adds a special feature BOS (Beginning Of Sentence).
If the current word is not the last in the sentence (i < len(sent)-1), it adds features for the next word (+1:) with similar information as mentioned above.
If the current word is the last in the sentence (i == len(sent)-1), it adds a special feature EOS (End Of Sentence).
"""


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
X_train = [sent2features(s) for s in training]
X_test = [sent2features(s) for s in test]

def sent2labels(sent):
    return [label for token, postag, label in sent]
y_train = [sent2labels(s) for s in training]
y_test = [sent2labels(s) for s in test]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,#L1 regularization
    c2=0.1,#L2 regularization
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)
""""
X_train=[("The", "DT", "B-NP"), ("dog", "NN", "I-NP"), ("barks", "VBZ", "O")]
Y_train=[("A", "DT", "B-NP"), ("cat", "NN", "I-NP"), ("meows", "VBZ", "O")]
"""


fig, ax = plt.subplots(figsize=(30,10))
font = {'family' : 'sans-serif',
        'size'   : 16}
matplotlib.rc('font', **font)

final_text = []
color = []
samples = 10
integer = random.randint(0,500)
prediction = crf.predict(X_test[integer:integer+samples])
for x,y in zip(test[integer:integer+samples], prediction):
    for x1,y1, in zip(x,y):
        if y1!='O':
            final_text.append('<{}>'.format(x1[0]))
            if y1[0]=='I':
                color.append(color[-1])
            else:
                  color.append ({'color':random.choice(['blue','green','red','magenta'])})
        else:
            final_text.append(x1[0])
    final_text.append('\n')
# You can either create a HighlightText object
HighlightText(x=0, y=0.75,
              s=' '.join(final_text),
              highlight_textprops=color,
              ax=ax)
plt.axis('off')

