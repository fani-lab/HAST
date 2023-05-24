import argparse
import os
import pickle
import json
import numpy as np
from random import random
import re

from cmn.review import Review


def load(reviews, splits):
    print('\n Loading reviews and preprocessing ...')
    print('#' * 50)
    try:
        print('\nLoading reviews file ...')
        with open(f'{reviews}', 'rb') as f:
            reviews = pickle.load(f)
        with open(f'{splits}', 'r') as f:
            splits = json.load(f)
    except (FileNotFoundError, EOFError) as e:
        print(e)
        print('\nLoading existing file failed!')
    print(f'(#reviews: {len(reviews)})')
    return reviews, splits


def preprocess(org_reviews, is_test):
    reviews_list = []
    for r in org_reviews:
        if not len(r.aos[0]):
            continue
        else:
            aos_list = []
            for aos_instance in r.aos[0]:
                aos_list.extend(aos_instance[0])
            # text = ' '.join(r.sentences[0]).replace('*****', '').replace('   ', '  ').replace('  ', ' ').replace('  ', ' ') + '####'
            text = re.sub(r'\s{2,}', ' ', ' '.join(r.sentences[0]).replace('*****', '').strip()) + '####'
            for idx, word in enumerate(r.sentences[0]):
                if is_test and word == "*****":
                    continue
                elif idx in aos_list:
                    tag = word + '=T' + ' '
                    text += tag
                else:
                    tag = word + '=O' + ' '
                    text += tag
            reviews_list.append(text.rstrip())
    return reviews_list


# python main.py -ds_name [YOUR_DATASET_NAME] -sgd_lr [YOUR_LEARNING_RATE_FOR_SGD] -win [YOUR_WINDOW_SIZE] -optimizer [YOUR_OPTIMIZER] -rnn_type [LSTM|GRU] -attention_type [bilinear|concat]
def main(args):
    if not os.path.isdir(f'{args.output}'): os.makedirs(f'{args.output}')
    org_reviews, splits = load(args.reviews, args.splits)
    test = np.array(org_reviews)[splits['test']].tolist()

    for h in range(0, 101, 10):

        path = f'{args.output}/{h}/{args.dname}'
        if not os.path.isdir(f'{args.output}/{h}'):
            os.makedirs(f'{args.output}/{h}')

        hp = h / 100
        test_hidden = []
        for t in range(len(test)):
            if random() < hp:
                test_hidden.append(test[t].hide_aspects())
            else:
                test_hidden.append(test[t])
        preprocessed_test = preprocess(test_hidden, True)

        with open(f'{path}_test.txt', 'w') as file:
            for d in preprocessed_test:
                file.write(d + '\n')
        with open(f'{path}_test_opi_ds.txt', 'w') as file:
            for i in range(len(preprocessed_test)):
                file.write('a +1' + '\n')

    train = preprocess(np.array(org_reviews)[splits['folds']['0']['train']].tolist(), False)
    path = f'{args.output}/{args.dname}'
    with open(f'{path}_train.txt', 'w') as file:
        for d in train:
            file.write(d + '\n')
    with open(f'{path}_train_opi_ds.txt', 'w') as file:
        for i in range(len(train)):
            file.write('a +1' + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HAST Wrapper')
    parser.add_argument('--dname', dest='dname', type=str, default='16semeval_rest')
    parser.add_argument('--reviews', dest='reviews', type=str,
                        default='../../output/HAST-Dataset/English/train/Semeval-2016/reviews.pkl',
                        help='raw dataset file path')
    parser.add_argument('--splits', dest='splits', type=str,
                        default='../../output/HAST-Dataset/English/train/Semeval-2016/splits.json',
                        help='raw dataset file path')
    parser.add_argument('--output', dest='output', type=str,
                        default='../../output/HAST-Dataset/English/train/Semeval-2016',
                        help='output path')
    args = parser.parse_args()

    main(args)
