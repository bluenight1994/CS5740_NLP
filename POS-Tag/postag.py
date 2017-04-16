
# coding: utf-8

import numpy as np
import csv
import collections

'''
Read data subroutine
'''

print "------- read train data ---------"
data = []
tags = []
with open('./data/train_x.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append(row['word'])
with open('./data/train_y.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        tags.append(row['tag'])
assert len(data) == len(tags) , "data not aligned with tags"

'''
Read dev data subroutine
'''
print "------- read dev data ---------"
dev_data = []
dev_tags = []
with open('./data/dev_x.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        dev_data.append(row['word'])
with open('./data/dev_y.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        dev_tags.append(row['tag'])

'''
part 1: MLE Parameter Estimation
ues a greedy decoder to find maximum scoring sequence
'''

print "part 1 MLE estimation simple count"
freq_map = {}
for word in data:
    if word not in freq_map:
        freq_map[word] = 1
    else:
        freq_map[word] += 1

mm = collections.defaultdict(list)
total_cnt = len(data)

for i in xrange(total_cnt):
    mm[data[i]].append(tags[i])

model = {}
for k in mm.keys():
    candidates = mm[k]
    cnt = collections.Counter(candidates)
    model[k] = cnt.most_common(1)[0][0]

res = 0
for i in range(len(dev_data)):
    cur = dev_data[i]
    if cur in model:
        if model[cur] == dev_tags[i]:
            res += 1

print "accuracy of naive counting(baseline)"
# base line accuracy without smoothing
print "accuracy:", float(res) / len(dev_data)


'''
training part:
maximum likelihood methods for estimating 
    1. transitions q
    2. emissions e
'''

# count occurence of tags
print "building model....."
tt = list(set(tags))
tt.append('START')
tt.append('STOP')
map_tt = {}
for i in range(len(tt)):
    map_tt[tt[i]] = i

cy = {}
for i in tags:
    if map_tt[i] not in cy: cy[map_tt[i]] = 1
    else: cy[map_tt[i]] += 1

cy[map_tt['START']] = 0
cy[map_tt['STOP']] = 0

ss = list(set(data))
map_ss = {}
for i in range(len(ss)):
    map_ss[ss[i]] = i

# nn : categories of tags
nn = len(tt)
# mm : categories of words
mm = len(ss)

# cyy: p(y[i-1],y[i]) 
cyy = np.zeros((nn, nn))
# cyyy: p(y[i-2], y[i-1], y[i])
cyyy = np.zeros((nn, nn, nn))
cyx = np.zeros((mm, nn))

for i in range(len(data)):
    _word = data[i]
    _tag = tags[i]
    cyx[map_ss[_word]][map_tt[_tag]] += 1


'''
bigram model part: fill in cyy
'''
print "building bigram model"
DOT = '.'

tmp = []

for i in range(len(tags)):
    if tags[i] == DOT:
        tmp.append(i)

_tags = tags[:]
prev = 0

for idd in tmp:
    sentence = _tags[prev:idd]
    prev = idd + 1
    cy[map_tt['START']] += 1
    cy[map_tt['STOP']] += 1
    sentence = ['START'] + sentence + ['STOP']
    for j in range(len(sentence)-1):
        l, r = map_tt[sentence[j]], map_tt[sentence[j+1]]
        cyy[l][r] += 1


'''
trigram model part: fill in cyyy: p(y[i-2], y[i-1], y[i])
'''
print "building trigram model"
prev = 0
for idd in tmp:
    sentence = _tags[prev:idd]
    prev = idd + 1
    sentence = ['START'] + sentence + ['STOP']
    for j in range(len(sentence)-2):
        l, m, r = map_tt[sentence[j]], map_tt[sentence[j+1]], map_tt[sentence[j+2]]
        cyyy[l][m][r] += 1



'''
Implementing The Viterbi Algorithm
Dynamic programming + Backtrack
without handle for unknown smoothing, bigram : 0.925492416547
'''

print "Viterbi :"
dot_id_tmp = []

for i in range(len(dev_tags)):
    if dev_tags[i] == DOT:
        dot_id_tmp.append(i) 

prev = 0
final = []
cnt = 0
ac = 0
bc = 0
        
for idd in dot_id_tmp:
    sentence_tag = dev_tags[prev: idd]
    sentence_word = dev_data[prev: idd]
    prev = idd + 1
    sentence_tag = sentence_tag
    sentence_word = ['START'] + sentence_word + ['STOP']
    # dp subroutine
    # initial parameters for dp
    N = len(sentence_word)
    k = len(tt)
    dp = np.zeros((k,N))
    track = np.zeros((k,N))
    # total time complexity Nk^2
    for i in range(1, N):
        cur_word = sentence_word[i]
        if cur_word in map_ss:
            word_id = map_ss[cur_word]
        else:
            # case for unknown word
            word_id = 0
        # j current loop idx
        for j in range(0, k):
            # s previous layer idx
            # emission = c(y,x) / c(y)
            e = np.log(float(cyx[word_id][j]) / cy[j])
            candidates = []
            for s in range(0, k):
                qq = float(cyy[s][j]) / cy[s]
                # add-k smoothing
                q = np.log((float(cyy[s][j]) + 1) / (cy[s] + 46))
                # linear interpolation 
                # q = np.log(0.99 * qq + 0.01 * cy[j])
                candidates.append(dp[s][i-1] + q)
            res = np.argmax(candidates)
            dp[j][i] = candidates[res] + e
            track[j][i] = res
    ret = []
    startIdx = np.argmax(dp[:,N-1])
    ret.append(startIdx)
    for m in range(N-1, 0, -1):
        ret.append(track[startIdx,m])
        startIdx = int(track[startIdx,m])
    ret = ret[::-1]
    # strip the added start and stop sign
    predict = [tt[int(uu)] for uu in ret[1:len(ret)-1]]
    ccnt = 0
    cnt += 1
    for u in range(len(predict)):
        if predict[u] == sentence_tag[u]:
            ccnt += 1
    ac += ccnt
    bc += len(predict)
    if cnt % 100 == 0:
        print float(ac) / bc
    final += predict + ["."]



# wrapper class for Trellis Algorithm
class state:
    def __init__(self):
        self.acc = []
        self.score = 0.0
        self.cur = None



'''
Implementing The Trellis Algorithm
Greedy top k

bigram k = 1 dev 0.9250232366
bigram k = 2 dev 0.924991031254
bigram k = 3 dev 0.924991031254
'''

print "trellis"
k = 3

prev = 0
ac = 0
bc = 0
cnt = 0

for idd in dot_id_tmp:
    sentence_tag = dev_tags[prev: idd]
    sentence_word = dev_data[prev: idd]
    prev = idd + 1
    sentence_tag = sentence_tag
    sentence_word = ['START'] + sentence_word + ['STOP']
    N = len(sentence_word)
    k = len(tt)
    dp = np.zeros((k,N))
    q = []
    init = state()
    init.cur = map_tt["START"]
    init.acc.append(map_tt["START"])
    q.append(init)
    for i in range(1, N):
        cur_word = sentence_word[i]
        if cur_word in map_ss:
            word_id = map_ss[cur_word]
        else:
            word_id = 0
        # loop through all the state in q
        tmp = []
        for ss in q:
            s = ss.cur    
            for j in range(0, k):
                e = np.log(float(cyx[word_id][j]) / cy[j])
                q = np.log(float(cyy[s][j]) / cy[s])
                ns = state()
                ns.cur = j
                ns.score = ss.score + e + q
                ns.acc = ss.acc + [j]
                tmp.append(ns)
        tmp.sort(key = lambda x: x.score, reverse = True)
        q = []
        q += tmp[:k]
    ret = []
    q.sort(key = lambda x: x.score, reverse = True)
    for uuu in q[0].acc:
        ret.append(tt[uuu])
    ret = ret[1:len(ret)-1]
    ccnt = 0
    for u in range(len(ret)):
        if ret[u] == sentence_tag[u]:
            ccnt += 1
    ac += ccnt
    bc += len(ret)
    cnt += 1
    if cnt % 100 == 0:
        print float(ac) / bc





