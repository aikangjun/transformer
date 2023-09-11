import random
import numpy as np
import tensorflow as tf

import pandas as pd


class Vocab():
    def __init__(self,
                 question_path,
                 answer_path):
        self.question_path = question_path
        self.answer_path = answer_path

    def random_dic(self, dicts):
        dict_key_ls = list(dicts.keys())
        random.seed(1)
        random.shuffle(dict_key_ls)
        new_dic = {}
        for key in dict_key_ls:
            new_dic[key] = dicts.get(key)
        return new_dic

    def create_vocab(self):
        question_df = pd.read_csv(self.question_path)
        answer_df = pd.read_csv(self.answer_path)
        question_df['wordlist'] = question_df.content.apply(lambda x: list(x))
        answer_df['wordlist'] = answer_df.content.apply(lambda x: list(x))
        word_counts = {}
        for q in question_df.wordlist:
            for word in q:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
        for a in answer_df.wordlist:
            for word in a:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1

        word_counts = dict(sorted(word_counts.items(),key=lambda x:x[1],reverse=True))
        print('正在生成vocab...')
        with open('..\\vocab\\vocab.txt', 'w', encoding='utf-8', errors='ignore') as f:
            f.write('[PAD]' + '\n' + '[START]' + '\n' + '[END]' + '\n' + '[UNK]' + '\n')
            for i in range(10):
                f.write(f'[unused{i}]' + '\n')
            for k, v in word_counts.items():
                f.write(str(k) + '\n')



if __name__ == '__main__':
    import configure.config as cfg

    vocab = Vocab(question_path=cfg.question_path,
                  answer_path=cfg.answer_path)
    vocab.create_vocab()

