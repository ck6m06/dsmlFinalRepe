#!/usr/bin/env python3
import json, random, os
from collections import Counter

def main():
    infile = r"d:\\dsmlFinalRepe\\Distinguishing_Ignorance_from_Error_in_LLM_Hallucinations\\experiment\\type2_results\\open_book_eval_results.json"
    out_prefix = r"d:\\dsmlFinalRepe\\Distinguishing_Ignorance_from_Error_in_LLM_Hallucinations\\experiment2\\datasets\\open_book_eval_results"
    seed = 42
    train_frac = 0.7
    val_frac = 0.15
    test_frac = 0.15

    random.seed(seed)
    with open(infile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = data['results'] if isinstance(data, dict) and 'results' in data else data

    true_idxs = [i for i, r in enumerate(records) if r.get('correct')]
    false_idxs = [i for i, r in enumerate(records) if not r.get('correct')]
    random.shuffle(true_idxs)
    random.shuffle(false_idxs)

    def split_list(idxs):
        n = len(idxs)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        train = idxs[:n_train]
        val = idxs[n_train:n_train + n_val]
        test = idxs[n_train + n_val:]
        return train, val, test

    t_tr, t_val, t_test = split_list(true_idxs)
    f_tr, f_val, f_test = split_list(false_idxs)
    train_idx = sorted(t_tr + f_tr)
    val_idx = sorted(t_val + f_val)
    test_idx = sorted(t_test + f_test)

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    train_recs = [records[i] for i in train_idx]
    val_recs = [records[i] for i in val_idx]
    test_recs = [records[i] for i in test_idx]

    with open(out_prefix + '_train.json', 'w', encoding='utf-8') as f:
        json.dump({'results': train_recs}, f, ensure_ascii=False, indent=2)
    with open(out_prefix + '_val.json', 'w', encoding='utf-8') as f:
        json.dump({'results': val_recs}, f, ensure_ascii=False, indent=2)
    with open(out_prefix + '_test.json', 'w', encoding='utf-8') as f:
        json.dump({'results': test_recs}, f, ensure_ascii=False, indent=2)
    with open(out_prefix + '_splits.json', 'w', encoding='utf-8') as f:
        json.dump({'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx, 'seed': seed}, f, indent=2)

    print('train', len(train_recs), Counter([r.get('correct') for r in train_recs]))
    print('val', len(val_recs), Counter([r.get('correct') for r in val_recs]))
    print('test', len(test_recs), Counter([r.get('correct') for r in test_recs]))

if __name__ == '__main__':
    main()
