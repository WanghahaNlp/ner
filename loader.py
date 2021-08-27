import os
import re
import codecs

from data_utils import create_dico, create_mapping, zero_digits
from data_utils import iob2, iob_iobes, get_seg_features


def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    num = 0
    for line in codecs.open(path, 'r', 'utf8'):
        num += 1
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
            else:
                word = line.split()
            assert len(word) >= 2, print([word[0]])
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    更新bio标注方案，兼容III->BII情况，生成BI->BE  BII->BIE
    Args:
        sentences(list): 数据集
        tag_scheme(str): 标注方案
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # 检查标注文件是否以IOB格式给出
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            # 转换成BIE的形式，BI->BE  BII->BIE
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def char_mapping(sentences, lower):
    """
    按词频排序创建单词映射字典。
    Args:
        sentences(list): 数据集
        lower(bool): 英文是否全部小写
    return:
        dico(dict): 词频统计
        tag_to_id(dict): 词: index
        id_to_tag(dict): index: 词
    """
    # 一条完整序列的集合
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    # 统计词频
    dico = create_dico(chars)
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    # 字符到id，id到字符{'海'：1}{1：海}
    char_to_id, id_to_char = create_mapping(dico)
    print("找到 %i 独特的单词 (总数 %i 个)" % (
        len(dico), sum(len(x) for x in chars)
    ))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    统计标签词频
    """
    tags = [[char[-1] for char in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("找到 %i 唯一命名的实体标签" % len([i for i in dico.keys() if "B" in i]))
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, train=True):
    """
    准备数据集。返回包含以下内容的字典列表列表:
    Args:
        sentences
        char_to_id
        tag_to_id
        lower
        train
    return:
        data(list): 每个序列的  【字list, 字index, 字123, bio的index】
    """

    none_index = tag_to_id["O"]

    def f(x):
        return x.lower() if lower else x

    data = []
    for s in sentences:
        string = [w[0] for w in s]
        chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>'] for w in string]
        # 数据转换成bies形式
        segs = get_seg_features("".join(string))
        if train:
            tags = [tag_to_id[w[-1]] for w in s]
        else:
            tags = [none_index for _ in chars]
        data.append([string, chars, segs, tags])

    return data


def augment_with_pretrained(dictionary, ext_emb_path, chars):
    """
    用预先训练过的嵌入词来扩充字典。
    如果“words”为None，我们添加每一个预先训练过的嵌入单词
    否则，我们只添加指定的单词
    “单词”(通常是开发和测试集中的单词)。
    Args:
        dictionary(dict): 词频字典
        ext_emb_path(str): 字向量相对路径
        chars(list): 测试序列
    """
    print('加载来自 %s 预训练的 embeddings...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(line.rstrip()) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    # 我们要么在预先训练好的文件中添加每个单词，
    # 或仅在“单词”列表中给出的单词
    # 我们可以分配一个预先训练好的嵌入
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            # 如果测试序列中字或词在预训练的向量中 和 字或词 不在训练集中 
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:
                # 在词典中添加 字:数量0
                dictionary[char] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def save_maps(save_path, *params):
    """
    Save mappings and invert mappings
    """
    pass
    # with codecs.open(save_path, "w", encoding="utf8") as f:
    #     pickle.dump(params, f)


def load_maps(save_path):
    """
    Load mappings from the file
    """
    pass
    # with codecs.open(save_path, "r", encoding="utf8") as f:
    #     pickle.load(save_path, f)
