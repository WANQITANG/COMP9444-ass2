import tensorflow as tf
# -*- coding:utf8 -*-
# import glob
# import os
import re
# import nltk.stem
# nltk.download()
# import unicodedata
# from nltk import word_tokenize
# from nltk.stem import LancasterStemmer,WordNetLemmatizer
BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 150  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})

def preprocess(review):##
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    review=review.lower()
    # print(review)
    words=review.split(" ")##seperate single item
    # print(words)
    processed_review = []
    for word in words:
        # new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_word = re.sub(r'\W', '', word)  ##remove punctuation
        if word not in stop_words:
            if len(word) >= 4:
                # print(len(word))
                ##remove stop-word
                processed_review.append(new_word)  ##Remove non-ASCII characters from list of tokenized words
    while '' in processed_review:
        processed_review.remove('')##Remove non-ASCII characters from list of tokenized words
    # print('',new_words)
    ##
    # stems=[]
    # for word in processed_review:
    #     stemmer=LancasterStemmer()
    #     stem=stemmer.stem(word)
    #     stems.append(stem)
    # # print('',stems)
    # ##
    # lemmatize_verb = []
    # for word in processed_review:
    #     lemmatizer= WordNetLemmatizer()
    #     lemma = lemmatizer.lemmatize(word,pos='v')
    #     lemmatize_verb.append(lemma)
    # print('', lemmatize_verb)

    # print(processed_review)
    return processed_review
# data=[]
# path= './data/train'
# dir = os.path.dirname(__file__)
# file_list = glob.glob(os.path.join(dir, './data/train'+ '/pos/*'))
# file_list.extend(glob.glob(os.path.join(dir,'./data/train' + '/neg/*')))
# print("Parsing %s files" % len(file_list))
# for i, f in enumerate(file_list):
#     with open(f, "r",encoding='utf-8') as openf:
#         s = openf.read()
#         ss = preprocess(s)
#         # print(ss)
#         data.append(ss)
#         break
# print(data)

def lstm_cell(NumUnits,dropout_keep_prob):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=NumUnits,forget_bias=1.0,state_is_tuple=True,activation=tf.tanh)
    drop_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm, output_keep_prob=dropout_keep_prob)
    # lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=NumUnits, forget_bias=1.0, state_is_tuple=True, activation=tf.tanh)
    # init_state_bw = lstm_cell_fw.zero_state(BATCH_SIZE, dtype=tf.float32)
    # drop_cell_bw = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell_bw, output_keep_prob=dropout_keep_prob)
    return drop_cell
def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """
##(128,100,50)
    input_data=tf.placeholder(tf.float32,[BATCH_SIZE,MAX_WORDS_IN_REVIEW,EMBEDDING_SIZE],name="input_data")##
    labels=tf.placeholder(tf.float32,[BATCH_SIZE,2],name='labels')
    # print(input_data,labels)
    dropout_keep_prob = tf.placeholder_with_default(0.6, shape = ())
    NumUnits = 64
    weights= tf.Variable(tf.truncated_normal([NumUnits,2],stddev=0.2))
    bias=tf.Variable(tf.constant(0.1,shape=[2]))
    # input_data=tf.reshape(input_data,[-1,MAX_WORDS_IN_REVIEW])
    # print('input',input_data)
    # X_data=tf.matmul(input_data,weights['in'])+bias['in']
    # print(X_data)
    # X_data=tf.reshape(X_data,[-1,MAX_WORDS_IN_REVIEW,NumUnits])
    # print(X_data)

    mlstm_cell=tf.contrib.rnn.MultiRNNCell([lstm_cell(NumUnits,dropout_keep_prob) for _ in range(2)],state_is_tuple=True)
    init_state_fw = mlstm_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
    # print(init_state_fw)
    # print(mlstm_cell)
    # print(input_data)

    outputs_fw ,outputs_state_fw = tf.nn.dynamic_rnn(cell=mlstm_cell,initial_state=init_state_fw,dtype=tf.float32, inputs=input_data,time_major=False)
    print(outputs_fw)
    # output=tf.concat((outputs_fw, outputs_bw),1)
    # print(output)
    # print((outputs_state_fw, outputs_state_bw))
    # outputs_fw=outputs_fw[:,:,-1]
    print('ddd',outputs_fw,labels)
    output = tf.transpose(outputs_fw, [1, 0, 2])
    print('output(transpose)',output,'get_shape[0]',output.get_shape()[0])
    last = tf.gather(output, int(output.get_shape()[0]) - 1)
    print('la',last)
    last_output = tf.matmul(last,weights)+bias
    # print('last_o',last_output)
    # (128,2)
    prediction = last_output
    print(prediction)
    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    Accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32), name='accuracy')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels), name='loss')
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)##0.001
    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss
