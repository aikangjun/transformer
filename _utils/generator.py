import numpy as np
import pandas as pd

class Generater():
    def __init__(self,
                 question_path: str,
                 anwser_path: str,
                 train_candidatas_path: str,
                 test_candidates_path: str,
                 batch_size: int,
                 vocab_path: str,
                 max_len: int):
        self.quetion_path = question_path
        self.anwser_path = anwser_path
        self.train_candidates_path = train_candidatas_path
        self.test_candidates_path = test_candidates_path
        self.batch_size = batch_size
        self.max_len = max_len

        self.vocab = self.txt2dict(vocab_path)
        self.get_train_and_test()

    def txt2dict(self, file_path):
        dict = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                if line[:-1] not in dict:
                    dict[f'{line[:-1]}'] = i
        return dict

    def get_train_and_test(self):
        question_df = pd.read_csv(self.quetion_path)
        answer_df = pd.read_csv(self.anwser_path)
        question_df['wordlist'] = question_df.content.apply(lambda x: list(x))
        answer_df['wordlist'] = answer_df.content.apply(lambda x: list(x))

        train_ids = pd.read_csv(self.train_candidates_path)
        test_ids = pd.read_csv(self.test_candidates_path)
        # drop_duplicates 丢弃重复'question_id',每个问题对应一个答案
        train_ids = train_ids.drop_duplicates('question_id')
        test_ids = test_ids.drop_duplicates('question_id')

        train_data = train_ids.merge(question_df[['question_id', 'wordlist']], on='question_id', how='left')
        train_data = train_data.merge(answer_df[['ans_id', 'wordlist']], left_on='pos_ans_id', right_on='ans_id')

        test_data = test_ids.merge(question_df[['question_id', 'wordlist']], on='question_id')
        test_data = test_data.merge(answer_df[['ans_id', 'wordlist']], on='ans_id', how='left')

        self.train_sources = np.array(train_data['wordlist_x'])
        self.train_targets = np.array(train_data['wordlist_y'])

        self.test_sources = np.array(test_data['wordlist_x'])
        self.test_targets = np.array(test_data['wordlist_y'])

    def padding(self, seq):
        PAD, START, END, UNK = '[PAD]', '[START]', '[END]', '[UNK]'
        if len(seq) < self.max_len:
            seq = seq + [END] + [PAD] * (self.max_len - len(seq) - 1)
        else:
            seq = seq[:self.max_len - 1] + [END]
        for i, token in enumerate(seq):
            if token not in self.vocab:
                seq[i] = UNK
        return seq

    def add_padding(self, seq):
        PAD, START, END, UNK = '[PAD]', '[START]', '[END]', '[UNK]'
        if len(seq) < self.max_len - 2:
            seq = seq + [END] + [PAD] * (self.max_len - 2 - len(seq))
            seq = [START] + seq
        else:
            seq = seq[:self.max_len - 2]
            seq = [START] + seq + [END]
        return seq

    def token2index(self, seq):
        for i, token in enumerate(seq):
            if token not in self.vocab:
                seq[i] = ['[UNKONW]']
            seq[i] = self.vocab[token]

        return seq

    def index2token(self, seq):
        seq = list(seq)
        for i, index in enumerate(seq):
            for k, v in self.vocab.items():
                if index == v:
                    seq[i] = k
        return seq

    def get_train_len(self):
        if not len(self.train_sources) % self.batch_size:
            return len(self.train_sources) // self.batch_size
        else:
            return len(self.train_sources) // self.batch_size + 1

    def get_test_len(self):
        if not len(self.test_sources) % self.batch_size:
            return len(self.test_sources) // self.batch_size
        else:
            return len(self.test_sources) // self.batch_size + 1

    def generate(self, training=True):
        if training:
            sources = self.train_sources
            targets = self.train_targets
            while True:
                srcs, lgts, tgts = [], [], []
                for i, (source, target) in enumerate(zip(sources, targets)):
                    source = self.padding(source)
                    logit = self.add_padding(target)
                    target = self.padding(target)

                    source = self.token2index(source)
                    logit = self.token2index(logit)
                    target = self.token2index(target)

                    srcs.append(source)
                    lgts.append(logit)
                    tgts.append(target)
                    if len(srcs) == self.batch_size or i == len(sources) - 1:
                        anno_sources = np.array(srcs.copy())
                        anno_logits = np.array(lgts.copy())
                        anno_targets = np.array(tgts.copy())

                        srcs.clear()
                        lgts.clear()
                        tgts.clear()
                        yield anno_sources, anno_logits, anno_targets


if __name__ == '__main__':
    import configure.config as cfg

    gen = Generater(question_path=cfg.question_path,
                    anwser_path=cfg.answer_path,
                    train_candidatas_path=cfg.train_candidates_path,
                    test_candidates_path=cfg.test_candidates_path,
                    batch_size=cfg.batch_size,
                    vocab_path=cfg.vocab_path,
                    max_len=cfg.max_len)
    train_gen = gen.generate()
    k = gen.get_train_len()
    a, b, c = next(train_gen)
