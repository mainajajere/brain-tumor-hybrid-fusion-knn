import os, glob, csv, random, math
root = 'data/images'
classes = ['glioma','meningioma','pituitary','notumor']
seed = 1337
random.seed(seed)

rows = []
for i, c in enumerate(classes):
files = []
for ext in ('.png', '.jpg', '.jpeg'):
files += glob.glob(os.path.join(root, c, f'*{ext}'))
random.shuffle(files)
n = len(files)
n_test = math.floor(0.20 * n)
n_val  = math.floor(0.16 * n)
n_train = n - n_test - n_val
train = files[:n_train]
val   = files[n_train:n_train+n_val]
test  = files[n_train+n_val:]
rows.append((i, c, train, val, test))

def write_csv(path, rows_list):
with open(path, 'w', newline='') as f:
w = csv.DictWriter(f, fieldnames=['path','label','class'])
w.writeheader()
for i,c,files in rows_list:
for p in files:
w.writerow({'path': p, 'label': i, 'class': c})

os.makedirs('results/splits', exist_ok=True)
write_csv('results/splits/train.csv', [(i,c,tr) for (i,c,tr,,) in rows])
write_csv('results/splits/val.csv',   [(i,c,va) for (i,c,,va,) in rows])
write_csv('results/splits/test.csv',  [(i,c,te) for (i,c,,,te) in rows])

counts
from collections import Counter
import pandas as pd
def counts(csv_path):
import pandas as pd
df = pd.read_csv(csv_path)
return df.groupby('class').size().reindex(classes).fillna(0).astype(int)

cnt = pd.DataFrame({'train': counts('results/splits/train.csv'),
'val':   counts('results/splits/val.csv'),
'test':  counts('results/splits/test.csv')})
cnt.to_csv('results/splits/counts.csv')
print(cnt)
