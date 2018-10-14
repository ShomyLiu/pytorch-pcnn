# -*- coding: utf-8 -*-

id2rel = {}
rel = file('../dataset/SemEval/relation2id.txt')
for line in rel:
    line = line.strip('\n').split()
    id2rel[line[0]] = line[1]
rel.close()

res = file('./sem_CNN_ONE_result.txt')
out = file('sem_res.txt', 'w')
for line in res:
    line = line.strip('\n').split('\t')
    out.write(line[0] + '\t' + id2rel[line[1]] + '\n')
out.close()
