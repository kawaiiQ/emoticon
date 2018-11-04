import tensorflow as tf
import re
import os
import random
import json
from wsgiref.simple_server import make_server

global_sess = None
global_input_feature = None
global_cls_id = None
global_vocabulary = None
global_cls_path = None


def load_model(sess):
    saver = tf.train.import_meta_graph('.model/net.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint(".model/"))


def load_data():
    with open('.model/voc_cls.json', 'r') as f:
        data = json.load(f)
        vocabulary = data['vocabulary']
        cls_path = data['cls_path']
    return vocabulary, cls_path


def get_feature(words, vocabulary):
    feature = []
    for word in vocabulary:
        feature.append(len(re.findall(word, words)))
    return [feature]


def predict(sess, in_tensor, cls_tensor, vocabulary, cls_path, sentence):
    out = sess.run([cls_tensor], feed_dict={
        in_tensor: get_feature(sentence, vocabulary)
    })
    return cls_path[int(out[0])]


def get_pic(sentence, sess, input_feature, cls_id, vocabulary, cls_path):
    path = predict(sess, input_feature, cls_id, vocabulary, cls_path, sentence)
    path = path.replace('\\', '/')

    file_list = os.listdir(path)
    file_list.remove('synonyms')
    index = random.randint(0, len(file_list)-1)
    return path + '/' + file_list[index]


def application(environ, start_response):
    global global_sess
    global global_input_feature
    global global_cls_id
    global global_vocabulary
    global global_cls_path
    try:
        request_body_size = int(environ.get('CONTENT_LENGTH', 0))
    except ValueError:
        request_body_size = 0
    request_body = environ['wsgi.input'].read(request_body_size)
    request_body = request_body.decode('utf-8')
    request_body = json.loads(request_body)
    words = request_body['text'].strip(request_body['trigger_word']).strip()
    file_path = get_pic(words, global_sess,  global_input_feature, global_cls_id, global_vocabulary, global_cls_path)
    file_path = 'kawaiiq.xyz/emoticon/' + file_path
    start_response('200 OK', [('Content-Type', 'application/json')])
    print(file_path)
    body = {
      "attachments": [{
          "images": [
            {"url": "http://" + file_path},
          ]
      }]
    }
    return [json.dumps(body).encode('utf-8')]


def main():
    global global_sess
    global global_input_feature
    global global_cls_id
    global global_vocabulary
    global global_cls_path

    vocabulary, cls_path = load_data()
    sess = tf.Session()
    load_model(sess)
    graph = tf.get_default_graph()
    input_feature = graph.get_tensor_by_name('features:0')
    cls_id = graph.get_tensor_by_name('cls_id:0')

    global_sess = sess
    global_input_feature = input_feature
    global_cls_id = cls_id
    global_vocabulary = vocabulary
    global_cls_path = cls_path

    httpd = make_server('', 8000, application)
    print('Serving HTTP on port 8000...')
    # 开始监听HTTP请求:
    httpd.serve_forever()
    # ret = get_pic("你有病", sess,  input_feature, cls_id, vocabulary, cls_path)


if __name__ == '__main__':
    main()
