import torch


def id2mfconcept(idx, id2concept):
    mf = ['authority', 'care', 'fairness', 'loyalty', 'sanctity'][idx % 5]
    concept = idx // 5
    return mf, id2concept[concept]


def to_probs(w):
    with torch.no_grad():
        w = w.abs() / w.abs().sum()
    return w


def get_best(file, k=0):
    try:
        results = list()
        with open(file, 'r') as f:
            for l in f:
                if l.strip() == '':
                    continue
                if int(l.strip().split()[6]) >= k:
                    results.append(
                        (float(l.strip().split()[2]), float(l.strip().split()[3]), int(l.strip().split()[6]))
                    )
        return max(results)
    except (FileNotFoundError, ValueError):
        return None, None, None


def get_best_full(file, k=0):
    try:
        results = list()
        with open(file, 'r') as f:
            for l in f:
                if l.strip() == '':
                    continue
                if int(l.strip().split()[6]) >= k:
                    results.append((
                        float(l.strip().split()[2]),
                        float(l.strip().split()[3]),
                        float(l.strip().split()[4]),
                        float(l.strip().split()[5]),
                        int(l.strip().split()[6])
                    ))
        return max(results)
    except (FileNotFoundError, ValueError):
        return None, None, None
