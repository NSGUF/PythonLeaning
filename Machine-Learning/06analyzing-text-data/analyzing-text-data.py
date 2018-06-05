# -*- coding: utf-8 -*-
"""
@Created on 2018/2/24 13:39

@author: ZhifengFang
"""

words = ['table', 'probably', 'wolves', 'playing', 'is',
         'dog', 'the', 'beaches', 'grounded', 'dreamt', 'envision']
import numpy as np

import nltk

from nltk.corpus import brown
# 句子解析器
'''
text = "Are you curious about tokenization? Let's see how it works! We need to analyze a couple of sentences with punctuations to see it in action."

import nltk
from nltk.tokenize import sent_tokenize
# nltk.download()
# 提取标记
print(sent_tokenize(text))

'''
# 单词解析器
'''
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer

print(word_tokenize(text))# 基本单词解析器，单词内的标点符号不作分割
print(WordPunctTokenizer().tokenize(text))# 单词内的标点符号作分割

'''
# 题干提取方法
'''
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer



stemmers = ['PORTER', 'LANCASTER', 'SNOWBALL']

stemmer_porter = PorterStemmer()
stemmer_lancaster = LancasterStemmer()
stemmer_snowball = SnowballStemmer('english')

# 设置表格
formatted_row = '{:>10}' * (len(stemmers) + 1)
print('\n', formatted_row.format('WORD', *stemmers), '\n')
for word in words:
    stemmed_words = [stemmer_porter.stem(word),
                     stemmer_lancaster.stem(word), stemmer_snowball.stem(word)]  # 分别打印出三个算法结果
    print(formatted_row.format(word, *stemmed_words))
'''
# 词形还原
'''
from nltk.stem import WordNetLemmatizer
wordnet = WordNetLemmatizer()
lemmatizers = ['NOUN LEMMATIZER', 'VERB LEMMATIZER']
formatted_row = '{:>20}' * (len(lemmatizers) + 1)
print('\n', formatted_row.format('WORD', *lemmatizers), '\n')
for word in words:
    lemmatizer_words = [wordnet.lemmatize(word, pos='n'), wordnet.lemmatize(word, pos='v')]  # NOUN词形和VERB词形
    print(formatted_row.format(word, *lemmatizer_words))

'''
# 分块
#将文本分割成块
def splitter(data, num_words):
    # 定义需使用变量
    data = data.split(' ')
    output = []
    cur_count = 0
    cur_word = []
    # 迭代
    for word in data:
        cur_word.append(word)
        cur_count = cur_count + 1
        if cur_count == num_words:
            output.append(' '.join(cur_word))
            cur_word = []
            cur_count = 0
    output.append(' '.join(cur_word))
    return output
'''

if __name__=='__main__':
    print(brown.words()[:10000])
    data=' '.join(brown.words()[:10000])# 加载数据
    num_words=1700
    chunks=splitter(data,num_words)
    print(len(chunks))
    print(chunks)
'''
# 词袋模型
'''
if __name__ == '__main__':
    # 创建字典
    data = ' '.join(brown.words()[:10000])# 获取数据
    counter = 0
    chunks = []
    num_words = 2000
    text_chunks = splitter(data, num_words)# 分成5段
    for text in text_chunks:# 遍历每一段
        chunk = {'index': counter, 'text': text}
        chunks.append(chunk)# 分成5行2000列的数组
        counter += 1
    # 提取文档-词矩阵
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(min_df=5, max_df=.95)
    doc_term_matrix = vec.fit_transform([chunk['text'] for chunk in chunks])
    vocab = np.array(vec.get_feature_names())
    print(vocab)
    chunk_names = ['Chunk-0', 'Chunk-1', 'Chunk-2', 'Chunk-3', 'Chunk-4']
    formatted_row = '{:>12}' * (len(chunk_names) + 1)
    print(formatted_row.format('Word', *chunk_names), '\n')
    for word, item in zip(vocab, doc_term_matrix.T):
        output = [str(x) for x in item.data]
        print(formatted_row.format(word, *output))
'''
# 创建文本分类器
'''
from sklearn.datasets import fetch_20newsgroups

category_map = {'misc.forsale': 'Sales', 'rec.motorcycles': 'Motorcycles',
        'rec.sport.baseball': 'Baseball', 'sci.crypt': 'Cryptography',
        'sci.space': 'Space'}
print(category_map.keys())
training_data=fetch_20newsgroups(subset='train',
                               categories=category_map.keys(),shuffle=True,random_state=7)# 加载训练数据
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
X_train_termcounts=vectorizer.fit_transform(training_data.data)
print(X_train_termcounts.shape)
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
input_data = [
    "The curveballs of right handed pitchers tend to curve to the left",
    "Caesar cipher is an ancient form of encryption",
    "This two-wheeler is really good on slippery roads"
]
tfidf_transformer=TfidfTransformer()
X_train_tfidf=tfidf_transformer.fit_transform(X_train_termcounts)
classifier = MultinomialNB().fit(X_train_tfidf, training_data.target)
X_input_termcounts = vectorizer.transform(input_data)
X_input_tfidf = tfidf_transformer.transform(X_input_termcounts)
predicted_categories = classifier.predict(X_input_tfidf)
for sentence, category in zip(input_data, predicted_categories):
    print('\nInput:', sentence, '\nPredicted category:', \
            category_map[training_data.target_names[category]])
'''
# 识别性别
'''
import random
from nltk.corpus import names
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy

def gender_features(word, num_letters=2):# 提取输入单词的特征
    return {'feature': word[-num_letters:].lower()}

if __name__=='__main__':
    labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
            [(name, 'female') for name in names.words('female.txt')])# 获取数据，将男女数据合并到一个变量

    random.seed(7)
    random.shuffle(labeled_names)# 打乱顺序
    input_names = ['Leonardo', 'Amy', 'Sam']

    for i in range(1, 5):
        print(i)
        featuresets = [(gender_features(n, i), gender) for (n, gender) in labeled_names]# 获取单词特征
        train_set, test_set = featuresets[500:], featuresets[:500]# 分训练街和测试集
        classifier = NaiveBayesClassifier.train(train_set)#训练数据

        print('Accuracy ==>', str(100 * nltk_accuracy(classifier, test_set)) + str('%'))# 获取准确率

        for name in input_names:
            print(name, '==>', classifier.classify(gender_features(name, i)))# 计算结果

'''
# 分析句子的情感

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

def extract_features(word_list):
    return dict([(word, True) for word in word_list])

if __name__ == '__main__':
    positive_fileids = movie_reviews.fileids('pos')
    negative_fileids = movie_reviews.fileids('neg')
    print(positive_fileids)
    features_positive = [(extract_features(movie_reviews.words(fileids=[f])),
                          'Positive') for f in positive_fileids]
    features_negative = [(extract_features(movie_reviews.words(fileids=[f])),
                          'Negative') for f in negative_fileids]
    print(features_positive)

    threshold_factor = 0.8# 分训练数据集和测试数据集
    threshold_positive = int(threshold_factor * len(features_positive))
    threshold_negative = int(threshold_factor * len(features_negative))

    features_train = features_positive[:threshold_positive] + features_negative[:threshold_negative]
    features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]
    print(len(features_train))
    print("Number of test datapoints:", len(features_test))

    # Train a Naive Bayes classifier
    classifier = NaiveBayesClassifier.train(features_train)
    print("\nAccuracy of the classifier:", nltk.classify.util.accuracy(classifier, features_test))

    print("\nTop 10 most informative words:")
    for item in classifier.most_informative_features()[:10]:
        print(item[0])

    # Sample input reviews
    input_reviews = [
        "It is an amazing movie",
        "This is a dull movie. I would never recommend it to anyone.",
        "The cinematography is pretty great in this movie",
        "The direction was terrible and the story was all over the place"
    ]

    print("\nPredictions:")
    for review in input_reviews:
        print("\nReview:", review)
        probdist = classifier.prob_classify(extract_features(review.split()))
        pred_sentiment = probdist.max()
        print("Predicted sentiment:", pred_sentiment)
        print("Probability:", round(probdist.prob(pred_sentiment), 2))

