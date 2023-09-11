# text_preprocess\generator
answer_path = r'D:\dataset\nlp_language\cMedQA2-master\answer.csv'
question_path = r'D:\dataset\nlp_language\cMedQA2-master\question.csv'
train_candidates_path = r'D:\dataset\nlp_language\cMedQA2-master\train_candidates.txt'
test_candidates_path = r'D:\dataset\nlp_language\cMedQA2-master\test_candidates.txt'
vocab_path = r'C:\Users\chen\Desktop\zvan\transformer\vocab\vocab.txt'
batch_size = 8
max_len = 150
# custom/customlayers/network
num_heads = 8
embedding_size = 256
ffn_num_hidden = 256
ffn_num_outputs = 256
num_block = 6
dropout = 0.1
vocab_size = 4873
# train
cosine_schedule = True
ckpt_path = '.\\ckpt'
epochs = 30