import tensorflow as tf
from _utils.generator import Generater
import configure.config as cfg
from transformermodel import TransformerModel
import os

if __name__ == '__main__':
    gen = Generater(question_path=cfg.question_path,
                    anwser_path=cfg.answer_path,
                    train_candidatas_path=cfg.train_candidates_path,
                    test_candidates_path=cfg.test_candidates_path,
                    batch_size=cfg.batch_size,
                    vocab_path=cfg.vocab_path,
                    max_len=cfg.max_len)
    model = TransformerModel(num_block=cfg.num_block,
                             src_vocab_size=cfg.vocab_size,
                             tgt_vocab_size=cfg.vocab_size,
                             embedding_size=cfg.embedding_size,
                             num_heads=cfg.num_heads,
                             dropout=cfg.dropout,
                             lr=1e-3)
    if not os.path.exists(cfg.ckpt_path):
        os.makedirs(cfg.ckpt_path)
    ckpt = tf.train.Checkpoint(network=model.network,
                               optimizer=model.optimizer)
    ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt,
                                              directory=cfg.ckpt_path,
                                              max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print('最新检测点已加载！')

    train_gen = gen.generate()
    for epoch in range(cfg.epochs):
        print("start training...")
        for i in range(gen.get_train_len()):
            sources, logits, targets = next(train_gen)
            model.train(sources, logits, targets)
            if (i + 1) % 200 == 0:
                print(f'Batch: {i + 1}\t'
                      f'train_loss: {model.train_loss.result()}\t'
                      f'train_acc: {model.train_acc.result() * 100}\n')
                # for j in range(len(sources)):
                #     question = gen.index2token(sources[j])
                #     question = ''.join(question)
                #     answer = model.predict_seq2seq(sources[j], num_step=150, gen=gen)
                #     print(f'question: {question}\n'
                #           f'answer: {answer}\n')

        print(f'Epoch: {epoch + 1}\n'
              f'train_loss: {model.train_loss.result()}\n'
              f'train_acc: {model.train_acc.result() * 100}\n')

        save_path = ckpt_manager.save()
        model.train_loss.reset_state()
        model.train_acc.reset_state()

        for j in range(len(sources)):
            question = gen.index2token(sources[j])
            question = ''.join(question)
            answer = model.predict_seq2seq(sources[j], num_step=150, gen=gen)
            print(f'question: {question}\n'
                  f'answer: {answer}\n')
