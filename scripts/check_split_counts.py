import argparse, yaml
from collections import Counter
from src.data.dataset import list_images, make_splits

def main(cfg_path):
    with open(cfg_path) as f:
        C = yaml.safe_load(f)
    paths, labels = list_images(C['data']['root_dir'], C['data']['classes'])
    (tr_p,tr_y),(va_p,va_y),(te_p,te_y) = make_splits(paths, labels, C['data']['split'], C['data']['seed'])
    def counts(y): 
        c = Counter(y)
        return [c[i] for i in range(len(C['data']['classes']))]
    print('Classes:', C['data']['classes'])
    print('Train counts:', counts(tr_y), 'Total:', len(tr_y))
    print('Val counts:', counts(va_y), 'Total:', len(va_y))
    print('Test counts:', counts(te_y), 'Total:', len(te_y))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args.config)
