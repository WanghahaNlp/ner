# encoding=utf8
import os
import codecs
import pickle
import itertools
from collections import OrderedDict

import tensorflow as tf
import numpy as np
from model import Model
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, clean, create_model, save_model
from utils import print_config, save_config, load_config, test_ner
from data_utils import load_word2vec, create_input, input_from_line, BatchManager
import os

flags = tf.app.flags
flags.DEFINE_boolean("clean", True, "clean train folder")  # 删除训练的文件夹
flags.DEFINE_boolean("train", True, "Whether train the model")  # 是否训练模型
# configurations for the model 模型配置
flags.DEFINE_integer("seg_dim", 20, "Embedding size for segmentation, 0 if not used")  # embedding size为分割，0如果不使用
flags.DEFINE_integer("char_dim", 100, "Embedding size for characters")  # embedding size 为 字符集
flags.DEFINE_integer("lstm_dim", 100,
                     "Num of hidden units in LSTM, or num of filters in IDCNN")  # LSTM中隐藏单元的Num，或IDCNN中过滤器的Num
flags.DEFINE_string("tag_schema", "iobes", "tagging schema iobes or iob")  # 标记iobes或iob模式

# configurations for training
flags.DEFINE_float("clip", 5, "Gradient clip")  # 梯度剪辑
flags.DEFINE_float("dropout", 0.5, "Dropout rate")
flags.DEFINE_float("batch_size", 20, "batch size")
flags.DEFINE_float("lr", 0.001, "Initial learning rate")  # 最初的学习率
flags.DEFINE_string("optimizer", "adam", "Optimizer for training")  # 优化器
flags.DEFINE_boolean("pre_emb", True, "Wither use pre-trained embedding")  # 使用预训练嵌入
flags.DEFINE_boolean("zeros", False, "Wither replace digits with zero")  # 将数字替换为零
flags.DEFINE_boolean("lower", True, "Wither lower case")  # 全部小写

flags.DEFINE_integer("max_epoch", 100, "maximum training epochs")  # 最大的训练时期
flags.DEFINE_integer("steps_check", 100, "steps per checkpoint")  # 每个检查点的步骤
flags.DEFINE_string("ckpt_path", "ckpt", "Path to save model")  # 保存模型路径
flags.DEFINE_string("summary_path", "summary", "Path to store summaries")  # 存储摘要的路径
flags.DEFINE_string("log_file", "train.log", "File for log")  # 文件日志
flags.DEFINE_string("map_file", "maps.pkl", "file for maps")  # 文件映射
flags.DEFINE_string("vocab_file", "vocab.json", "File for vocab")  # 文件词汇
flags.DEFINE_string("config_file", "config_file", "File for config")  # 文件配置文件
flags.DEFINE_string("script", "conlleval", "evaluation script")  # 评估脚本
flags.DEFINE_string("result_path", "result", "Path for results")  # 路径为结果
flags.DEFINE_string("emb_file", os.path.join("data", "vec.txt"), "Path for pre_trained embedding")  # pre_training嵌入的路径
flags.DEFINE_string("train_file", os.path.join("data", "example.train"), "Path for train data")  # 数据集路径
flags.DEFINE_string("dev_file", os.path.join("data", "example.dev"), "Path for dev data")
flags.DEFINE_string("test_file", os.path.join("data", "example.test"), "Path for test data")
# flags.DEFINE_string("model_type", "idcnn", "Model type, can be idcnn or bilstm")  # 模型类型，可以是idcnn或bilstm
flags.DEFINE_string("model_type", "bilstm", "Model type, can be idcnn or bilstm")

FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"  # 渐变剪辑不应该太多
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"  # 随机失活0-1
assert FLAGS.lr > 0, "learning rate must larger than zero"  # 学习率必须大于0
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]  # 优化器


# config for the model
def config_model(char_to_id, tag_to_id):
    config = OrderedDict()
    config["model_type"] = FLAGS.model_type
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config


def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def train():
    # [[['海', 'O'], ['钓', 'O'], ['厦', 'B-LOC'], ['门', 'I-LOC'], ['与', 'O'], ...], ...]
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)  # 加载数据集，全部小写，数字不替换为0
    dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)

    # 数据处理成标准BIE形式
    update_tag_scheme(train_sentences, FLAGS.tag_schema)
    update_tag_scheme(test_sentences, FLAGS.tag_schema)

    # 如果不存在，创建文件映射
    if not os.path.isfile(FLAGS.map_file):
        # 是否使用预训练创建字典
        if FLAGS.pre_emb:
            # 训练集词频字典
            dico_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]
            # 测试集+训练集字典
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                FLAGS.emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])
                )
            )
        else:
            _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)

        # 为标记创建字典和映射
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        # 将程序中运行的对象信息保存到文件中去
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    # prepare data, get a collection of list containing index序列处理，生成相应的序列【字list, 字index, 字123, bio的index】
    train_data = prepare_dataset(
        train_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    print("%i / %i / %i 句子来自 train / dev / test." % (
        len(train_data), len(dev_data), len(test_data)))

    train_manager = BatchManager(train_data, FLAGS.batch_size)
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)
    # make path for store log and model if not exist
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(char_to_id, tag_to_id)
        save_config(config, FLAGS.config_file)

    log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        logger.info("start training")
        loss = []
        for i in range(100):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(
                        iteration, step % steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
            if best:
                save_model(sess, model, FLAGS.ckpt_path, logger)
            evaluate(sess, model, "test", test_manager, id_to_tag, logger)


def evaluate_line():
    # 加载配置文件
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger, False)
        while True:
            # try:
            #     line = input("请输入测试句子:")
            #     result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            #     print(result)
            # except Exception as e:
            #     logger.info(e)

            line = input("请输入测试句子:")
            result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            print(result)


def main(_):
    if FLAGS.train:
        if FLAGS.clean:
            clean(FLAGS)
        train()
    else:
        evaluate_line()


if __name__ == "__main__":
    tf.app.run(main)
