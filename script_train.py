import os
import re
import jieba
import random
import tensorflow as tf
import json


def add_to_vocabulary(vocabulary, word_list):
    for word in word_list:
        vocabulary.append(word)


def cut_words(sentence):
    if sentence[-1] == ')':
        sentence = sentence[:-4]
    words = list(set(jieba.cut(sentence, cut_all=True)))
    while '' in words:
        words.remove('')
    words.append(sentence)
    return words


def scan_image(path, vocabulary, cls_path, pic_data):
    words_list = []
    for item in os.listdir(path):
        new_path = os.path.join(path, item)
        if os.path.isdir(new_path):
            scan_image(new_path, vocabulary, cls_path, pic_data)
        elif item == 'synonyms':
            with open(new_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    synonym = line.strip()
                    if synonym != '':
                        words_list.append(synonym)
        else:
            words = item.split('.')[0]
            words_list.append(words)
    if len(words_list) != 0:
        cls_id = len(cls_path)
        cls_path.append(path)
        for words in words_list:
            add_to_vocabulary(vocabulary, cut_words(words))
            pic_data['words'].append(words)
            pic_data['labels'].append(cls_id)


def transform_vocabulary(vocabulary):
    vocabulary = list(set(vocabulary))
    for i in range(len(vocabulary)):
        if '.' in vocabulary[i]:
            old = vocabulary[i]
            new = ''
            for c in old:
                if c == '.':
                    new += '\\.'
                else:
                    new += c
            vocabulary[i] = new
    return vocabulary


def get_feature(words, vocabulary):
    feature = []
    for word in vocabulary:
        feature.append(len(re.findall(word, words)))
    return feature


def generate_features(pic_data, vocabulary):
    for words in pic_data['words']:
        pic_data['features'].append(get_feature(words, vocabulary))


def build_graph(num_words, num_cls):
    print('building graph ...')
    in_feature = tf.placeholder(dtype=tf.float32, shape=[None, num_words],
                                name='features')
    in_label = tf.placeholder(dtype=tf.int32, shape=[None],
                              name='labels')

    x = tf.layers.Dense(8192, activation=tf.nn.relu)(in_feature)
    x = tf.layers.Dense(4096, activation=tf.nn.relu)(x)
    x = tf.layers.Dense(2048, activation=tf.nn.relu)(x)
    x = tf.layers.Dense(1024, activation=tf.nn.relu)(x)
    x = tf.layers.Dense(512, activation=tf.nn.relu)(x)
    x = tf.layers.Dense(num_cls)(x)

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.01
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.9, staircase=True)

    train_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=x, labels=tf.one_hot(in_label, depth=num_cls))
    train_loss = tf.reduce_mean(train_loss, name='loss')
    tf.train.GradientDescentOptimizer(learning_rate).minimize(train_loss, name='train_op')

    out_prob = tf.nn.softmax(x, axis=1, name='cls_prob')
    tf.argmax(out_prob, axis=1, name='cls_id')


def load_data():
    print('scanning data ...')
    dir_list = []
    for item in os.listdir('.'):
        if item[0] != '.' and os.path.isdir(item):
            dir_list.append(item)

    vocabulary = []
    cls_path = []
    pic_data = {'words': [], 'features': [], 'labels': []}
    for dir_it in dir_list:
        scan_image(dir_it, vocabulary, cls_path, pic_data)
    print('generating vocabulary ...')
    transform_vocabulary(vocabulary)
    print('generating features ...')
    generate_features(pic_data, vocabulary)

    return vocabulary, cls_path, pic_data


def save_model(sess, saver, vocabulary, cls_path, training_data, step):
    saver.save(sess, '.model/net.ckpt')
    with open('.model/voc_cls.json', 'w') as f:
        json.dump({
            "vocabulary": vocabulary,
            "cls_path": cls_path,
            "step": step
        }, f)
    with open('.model/pic_data.json', 'w') as f:
        json.dump(training_data, f)


def train(sess, vocabulary, cls_path, training_data, step=1):
    features = training_data['features']
    labels = training_data['labels']
    num_data = len(features)

    epochs = 1500
    print(num_data, 'pictures')
    count = step - 1

    graph = tf.get_default_graph()
    input_feature = graph.get_tensor_by_name('features:0')
    input_label = graph.get_tensor_by_name('labels:0')
    cls_loss = graph.get_tensor_by_name('loss:0')
    run_train_op = graph.get_operation_by_name('train_op')
    saver = tf.train.Saver(max_to_keep=5)

    print_every = 10
    save_every = 500

    for epoch in range(epochs):
        order = list(range(num_data))
        random.shuffle(order)
        batch_size = 50
        while len(order) != 0:
            feature, label = [], []
            for _ in range(batch_size):
                index = order.pop()
                feature.append(features[index])
                label.append(labels[index])
                if len(order) == 0:
                    break

            feed_dict = {input_feature: feature, input_label: label}
            sess.run(run_train_op, feed_dict=feed_dict)

            if count % print_every == 0:
                print('epoch %s, step %s: loss is %s' % (epoch, count + 1, sess.run(cls_loss, feed_dict=feed_dict)))
            if count % save_every == 0:
                save_model(sess, saver, vocabulary, cls_path, training_data, step)
            count += 1


def main():
    if not os.path.exists('.model/checkpoint'):
        vocabulary, cls_path, pic_data = load_data()
        build_graph(len(vocabulary), len(cls_path))
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        train(session, vocabulary, cls_path, pic_data)


if __name__ == '__main__':
    main()
