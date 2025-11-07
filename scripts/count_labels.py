from collections import Counter
import os
p=os.path.dirname(os.path.dirname(__file__))
train_file=os.path.join(p,'trainlist03.txt')
test_file=os.path.join(p,'testlist03.txt')

with open(train_file,'r',encoding='utf-8') as f:
    train=[l.strip() for l in f if l.strip()]
with open(test_file,'r',encoding='utf-8') as f:
    test=[l.strip() for l in f if l.strip()]

def counts(lst):
    c=Counter()
    for path in lst:
        label=path.split('/')[0]
        c[label]+=1
    return c

print('TRAIN COUNTS')
for k,v in counts(train).most_common():
    print(f'{k}: {v}')
print('\nTEST COUNTS')
for k,v in counts(test).most_common():
    print(f'{k}: {v}')
