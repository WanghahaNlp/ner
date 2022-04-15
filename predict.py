# encoding=utf-8

from program_entry import *


if __name__ == '__main__':
    if FLAGS.train:
        raise Exception("train is True")
    evaluate_line()