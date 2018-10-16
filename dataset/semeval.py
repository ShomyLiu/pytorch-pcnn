# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import os
import numpy as np
import string


class SEMData(Dataset):

    def __init__(self, root_path, train=True):
        if train:
            path = os.path.join(root_path, 'train/npy/')
            print('loading train data')
        else:
            path = os.path.join(root_path, 'test/npy/')
            print('loading test data')

        self.word_feautre = np.load(path + 'word_feautre.npy')
        self.lexical_feature = np.load(path + 'lexical_feature.npy')
        self.right_pf = np.load(path + 'right_pf.npy')
        self.left_pf = np.load(path + 'left_pf.npy')
        self.labels = np.load(path + 'labels.npy')
        self.x = list(zip(self.lexical_feature, self.word_feautre, self.left_pf, self.right_pf, self.labels))
        print('loading finish')

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)


class SEMLoad(object):
    '''
    load and preprocess data
    '''
    def __init__(self, root_path, train=True, max_len=98, limit=50):

        self.stoplists = set(string.punctuation)

        self.max_len = max_len
        self.limit = limit
        self.root_path = root_path
        self.train = train
        if self.train:
            print('train data:')
        else:
            print('test data:')

        self.rel_path = os.path.join(root_path, 'relation2id.txt')
        self.w2v_path = os.path.join(root_path, 'vector_50.txt')
        self.train_path = os.path.join(root_path, 'train.txt')
        self.vocab_path = os.path.join(root_path, 'vocab.txt')
        self.test_path = os.path.join(root_path, 'test.txt')

        print('loading start....')
        self.rel2id, self.id2rel = self.load_rel()
        self.w2v, self.word2id, self.id2word = self.load_w2v()

        if train:
            self.lexical_feature, sen_feature, self.labels = self.parse_sen(self.train_path)
        else:
            self.lexical_feature, sen_feature, self.labels = self.parse_sen(self.test_path)

        self.word_feautre, self.left_pf, self.right_pf = sen_feature
        print('loading finish')

    def save(self):
        if self.train:
            prefix = 'train'
        else:
            prefix = 'test'
        np.save(os.path.join(self.root_path, prefix, 'npy/word_feautre.npy'), self.word_feautre)
        np.save(os.path.join(self.root_path, prefix, 'npy/left_pf.npy'), self.left_pf)
        np.save(os.path.join(self.root_path, prefix, 'npy/right_pf.npy'), self.right_pf)
        np.save(os.path.join(self.root_path, prefix, 'npy/lexical_feature.npy'), self.lexical_feature)
        np.save(os.path.join(self.root_path, prefix, 'npy/labels.npy'), self.labels)
        np.save(os.path.join(self.root_path, prefix, 'npy/w2v.npy'), self.w2v)
        print('save finish!')

    def load_rel(self):
        '''
        load relations
        '''
        rels = [i.strip('\n').split() for i in open(self.rel_path)]
        rel2id = {j: int(i) for i, j in rels}
        id2rel = {int(i): j for i, j in rels}

        return rel2id, id2rel

    def load_w2v(self):
        '''
        reading from vec.bin
        add two extra tokens:
            : UNK for unkown tokens
            : BLANK for the max len sentence
        '''
        wordlist = []
        vecs = []

        w2v = open(self.w2v_path)
        for line in w2v:
            line = line.strip('\n').split()
            word = line[0]
            vec = list(map(float, line[1:]))
            wordlist.append(word)
            vecs.append(np.array(vec))

        # wordlist.append('UNK')
        # wordlist.append('BLANK')
        # vecs.append(np.random.normal(size=dim, loc=0, scale=0.05))
        # vecs.append(np.random.normal(size=dim, loc=0, scale=0.05))
        # vecs.append(np.zeros(dim))
        # vecs.append(np.zeros(dim))
        word2id = {j: i for i, j in enumerate(wordlist)}
        id2word = {i: j for i, j in enumerate(wordlist)}

        return np.array(vecs, dtype=np.float32), word2id, id2word

    def parse_sen(self, path):
        '''
        parse the records in data
        '''
        all_sens =[]
        all_labels =[]
        for line in open(path, 'r'):
            line = line.strip('\n').split(' ')
            sens = line[5:]
            rel = int(line[0])

            ent1 = (int(line[1]), int(line[2]))
            ent2 = (int(line[3]), int(line[4]))

            all_labels.append(rel)
            sens = list(map(lambda x: self.word2id.get(x, self.word2id['<PAD>']), sens))

            all_sens.append((ent1, ent2, sens))

        lexical_feature = self.get_lexical_feature(all_sens)
        sen_feature = self.get_sentence_feature(all_sens)

        return lexical_feature, sen_feature, all_labels

    def get_lexical_feature(self, sens):
        '''
        : noun1
        : noun2
        : left and right tokens of noun1
        : left and right tokens of noun2
        : # WordNet hypernyms
        '''

        lexical_feature = []
        for idx, sen in enumerate(sens):
            pos_e1, pos_e2, sen = sen
            left_e1 = self.get_left_word(pos_e1, sen)
            left_e2 = self.get_left_word(pos_e2, sen)
            right_e1 = self.get_right_word(pos_e1, sen)
            right_e2 = self.get_right_word(pos_e2, sen)
            lexical_feature.append([sen[pos_e1[0]], left_e1, right_e1, sen[pos_e2[0]], left_e2, right_e2])

        return lexical_feature

    def get_sentence_feature(self, sens):
        '''
        : word embedding
        : postion embedding
        return:
        sen list
        pos_left
        pos_right
        '''
        update_sens = []

        for sen in sens:
            pos_e1, pos_e2, sen = sen
            pos_left = []
            pos_right = []
            ori_len = len(sen)
            for idx in range(ori_len):
                p1 = self.get_pos_feature(idx - pos_e1[0])
                p2 = self.get_pos_feature(idx - pos_e2[0])
                pos_left.append(p1)
                pos_right.append(p2)
            if ori_len > self.max_len:
                sen = sen[: self.max_len]
                pos_left = pos_left[: self.max_len]
                pos_right = pos_left[: self.max_len]
            elif ori_len < self.max_len:
                sen.extend([self.word2id['<PAD>']] * (self.max_len - ori_len))
                pos_left.extend([self.limit * 2 + 2] * (self.max_len - ori_len))
                pos_right.extend([self.limit * 2 + 2] * (self.max_len - ori_len))

            update_sens.append([sen, pos_left, pos_right])

        return zip(*update_sens)

    def get_left_word(self, pos, sen):
        '''
        get the left word id of the token of position
        '''
        pos = pos[0]
        if pos > 0:
            return sen[pos - 1]
        else:
            # return sen[pos]
            return self.word2id['<PAD>']

    def get_right_word(self, pos, sen):
        '''
        get the right word id of the token of position
        '''
        pos = pos[1]
        if pos < len(sen) - 1:
            return sen[pos + 1]
        else:
            # return sen[pos]
            return self.word2id['<PAD>']

    def get_pos_feature(self, x):
        '''
        clip the postion range:
        : -limit ~ limit => 0 ~ limit * 2+2
        : -51 => 0
        : -50 => 1
        : 50 => 101
        : >50: 102
        '''
        if x < -self.limit:
            return 0
        if -self.limit <= x <= self.limit:
            return x + self.limit + 1
        if x > self.limit:
            return self.limit * 2 + 1


if __name__ == "__main__":
    data = SEMLoad('./dataset/SemEval/', train=True)
    print(len(data.word2id))
    data.save()
    data = SEMLoad('./dataset/SemEval/', train=False)
    data.save()
