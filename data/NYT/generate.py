import json
import numpy as np

def load_data(in_file, word_dict, rel_dict, out_file):
    with open(in_file, 'r') as f1, open(out_file, 'w') as f2:
        lines = f1.readlines()
        for line in lines:
            line = json.loads(line)
            lengths, sents, spos = line[0], line[1], line[2]#NYT
            for i in range(len(sents)):
                new_line = dict()
                tokens = [word_dict[i] for i in sents[i]]
                sent = ' '.join(tokens)
                new_line['text'] = sent
                triples = np.reshape(spos[i], (-1,3))
                relationMentions = []
                for triple in triples:
                    rel = dict()
                    rel['subject'] = tokens[triple[0]]
                    rel['object'] = tokens[triple[1]]
                    rel['predicate'] = rel_dict[triple[2]]
                    relationMentions.append(rel)
                new_line['spo_list'] = relationMentions
                f2.write(json.dumps(new_line) + '\n')

def generate_schemas():
    fw = open('all_schemas', 'w', encoding='utf-8')
    with open('raw_NYT/relations2id.json', encoding='utf-8') as f:
        rel2id = json.load(f)
        tempDict = {}
        for i, j in rel2id.items():
            tempDict['predicate'] = i
            l = json.dumps(tempDict, ensure_ascii=False)
            fw.write(l)
            fw.write('\n')
    fw.close()

if __name__ == '__main__':
    #生成all_schemas文件
    generate_schemas()

    #生成数据集
    file_name_list = ['raw_NYT/train.json', 'raw_NYT/valid.json', 'raw_NYT/test.json']
    output_list = ['train_data.json', 'valid_data.json', 'test_data.json']
    for file_name, output in zip(file_name_list, output_list):
        with open('raw_NYT/relations2id.json', 'r') as f1, open('raw_NYT/words2id.json', 'r') as f2:
            rel2id = json.load(f1)
            words2id = json.load(f2)
        rel_dict = {j:i for i,j in rel2id.items()}
        word_dict = {j:i for i,j in words2id.items()}
        load_data(file_name, word_dict, rel_dict, output)