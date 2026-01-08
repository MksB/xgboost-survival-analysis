# Exacter Greedy Split
#
# ○ einfache Version für klassische Regressionbäume
# ○ GBDT - kompatible Version, Gradienten und Hessians

import numpy as np

# -----------------------------
# 1) Klassischer exakter Split
# Minimiert die Summe der quadrierten Fehler (SSE) für y.
# Eingaben:
#   x: 1D-array mit Feature-Werten
#   y: 1D-array mit Zielwerten
#   min_samples_leaf: minimale Anzahl Beobachtungen pro Blatt
# Rückgabe:
#   best_split: dict mit keys: 'threshold', 'left_idx', 'right_idx', 'sse'
# -----------------------------
def best_split_regression(x, y, min_samples_leaf=1):
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.ndim == 1 and y.ndim == 1 and x.shape[0] == y.shape[0]

    # sortieren nach Feature
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    n = len(x)

    # mögliche Splitpunkte sind zwischen verschiedenen x-Werten
    # kumulative Summen für schnelle Berechnung von Varianz/SSE
    cumsum_y = np.cumsum(ys)
    cumsum_y2 = np.cumsum(ys * ys)

    best = {'threshold': None, 'left_idx': None, 'right_idx': None, 'sse': np.inf}

    # iterate possible split positions (split after position i => left: [0..i], right: [i+1..n-1])
    for i in range(min_samples_leaf - 1, n - min_samples_leaf):
        # we only consider splits where xs[i] != xs[i+1] (distinct feature values)
        if xs[i] == xs[i + 1]:
            continue

        left_count = i + 1
        right_count = n - left_count

        left_sum = cumsum_y[i]
        left_sum2 = cumsum_y2[i]
        right_sum = cumsum_y[-1] - left_sum
        right_sum2 = cumsum_y2[-1] - left_sum2

        # SSE = sum(y^2) - (sum(y)^2 / n)
        sse_left = left_sum2 - (left_sum * left_sum) / left_count
        sse_right = right_sum2 - (right_sum * right_sum) / right_count
        sse_total = sse_left + sse_right

        if sse_total < best['sse']:
            threshold = 0.5 * (xs[i] + xs[i + 1])
            best = {
                'threshold': threshold,
                'left_idx': order[:left_count],
                'right_idx': order[left_count:],
                'sse': sse_total
            }

    return best


# -----------------------------
# 2) GBDT/gbm-style exakter Split (Gradient/Hessian)
# Minimiert die "loss improvement" basierend auf Gradienten g und Hessians h.
# Eingaben:
#   x: 1D-array Feature-Werte
#   g: 1D-array Gradienten (first derivative of loss wrt prediction)
#   h: 1D-array Hessians (second derivative; often ones for squared error)
#   min_samples_leaf: minimale Anzahl Beobachtungen pro Blatt
#   lambda_l2: L2 regularization on leaf weight
# Rückgabe:
#   best_split: dict mit keys: 'threshold', 'left_idx', 'right_idx', 'gain'
# Hinweis: Gain-Berechnung entspricht dem üblichen Newton-Split-Score:
#       score = G^2 / (H + lambda)
#       gain = score_left + score_right - score_parent
# -----------------------------
def best_split_gbdt(x, g, h=None, min_samples_leaf=1, lambda_l2=1.0):
    x = np.asarray(x)
    g = np.asarray(g)
    if h is None:
        h = np.ones_like(g)
    h = np.asarray(h)
    assert x.ndim == 1 and g.ndim == 1 and h.ndim == 1 and x.shape[0] == g.shape[0] == h.shape[0]

    order = np.argsort(x)
    xs = x[order]
    gs = g[order]
    hs = h[order]
    n = len(x)

    # kumulative Summen
    cumsum_g = np.cumsum(gs)
    cumsum_h = np.cumsum(hs)
    total_g = cumsum_g[-1]
    total_h = cumsum_h[-1]

    best = {'threshold': None, 'left_idx': None, 'right_idx': None, 'gain': -np.inf}

    # avoid division by zero; use same loop restrictions as above
    for i in range(min_samples_leaf - 1, n - min_samples_leaf):
        if xs[i] == xs[i + 1]:
            continue

        G_left = cumsum_g[i]
        H_left = cumsum_h[i]
        G_right = total_g - G_left
        H_right = total_h - H_left

        # score = G^2 / (H + lambda)
        score_left = (G_left * G_left) / (H_left + lambda_l2)
        score_right = (G_right * G_right) / (H_right + lambda_l2)
        score_parent = (total_g * total_g) / (total_h + lambda_l2)

        gain = score_left + score_right - score_parent

        if gain > best['gain']:
            threshold = 0.5 * (xs[i] + xs[i + 1])
            best = {
                'threshold': threshold,
                'left_idx': order[:i + 1],
                'right_idx': order[i + 1:],
                'gain': gain
            }

    return best


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    x = np.concatenate([rng.normal(0,1,50), rng.normal(5,1,50)])
    y = np.where(x>2.5,10+rng.normal(0,0.5,x.shape[0]),rng.normal(0,0.5,x.shape[0]))

    split = best_split_regression(x,y,min_samples_leaf=5)
    print("Regression best threshold:", split['threshold'],
          "SSE:", split['sse'])

    preds = np.zeros_like(y)
    g = preds - y
    h = np.ones_like(g)

    split_gbdt = best_split_gbdt(x,g,h,min_samples_leaf=5,
                                 lambda_l2=1.0)
    print("GBDT best threshold:", split_gbdt['threshold'],
          "Gain:", split_gbdt['gain'])
    
    
                                     
        

        
