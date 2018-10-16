# -*- coding: utf-8 -*-

data_dic ={
    'SEM': {
        'data_root': './dataset/SemEval/',
        'w2v_path': './dataset/SemEval/train/npy/w2v.npy',
        # 'p2v_path': './dataset/SemEval/train/npy/p2v.npy',
        'vocab_size': 22315,  # vocab + UNK + BLANK
        'rel_num': 19
    }
}


class DefaultConfig(object):

    model = 'PCNN'  # the name of used model, in  <models/__init__.py>
    data = 'SEM'  # SEM

    result_dir = './out/'
    data_root = data_dic[data]['data_root']  # the data dir
    w2v_path = data_dic[data]['w2v_path']
    # p1_2v_path = data_dic[data]['p1_2v_path']
    # p2_2v_path = data_dic[data]['p2_2v_path']
    load_model_path = 'checkpoints/model.pth'  # the trained model

    seed = 99
    batch_size = 128  # batch size
    use_gpu = True  # user GPU or not
    gpu_id = 0
    num_workers = 0  # how many workers for loading data

    max_len = 80 + 2  # max_len for each sentence + two padding
    limit = 50  # the position range <-limit, limit>

    # vocab_size = 114042 + 2  # vocab + UNK + BLANK
    vocab_size = data_dic[data]['vocab_size']  # vocab + UNK + BLANK
    rel_num = data_dic[data]['rel_num']
    word_dim = 50
    pos_dim = 5
    pos_size = limit * 2 + 2

    norm_emb=True

    num_epochs = 100  # the number of epochs for training
    drop_out = 0.5
    lr = 0.003  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0.  # optimizer parameter

    # Conv
    filters = [3]
    filters_num = 230
    sen_feature_dim = filters_num

    rel_dim = filters_num * len(filters)
    rel_filters_num = 100

    print_opt = 'DEF'
    use_pcnn=True


def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)
        data_list = ['data_root', 'w2v_path', 'rel_num', 'vocab_size']
        for r in data_list:
            setattr(self, r, data_dic[self.data][r])

        print('*************************************************')
        print('user config:')
        for k, v in kwargs.items():
            if not k.startswith('__'):
                print("{} => {}".format(k, getattr(self, k)))

        print('*************************************************')


DefaultConfig.parse = parse
opt =DefaultConfig()
