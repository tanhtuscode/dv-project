from collections import Counter
import os
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config.paths import config

# Use dynamic paths
train_file = str(config.TRAIN_LIST)
test_file = str(config.TEST_LIST)

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
