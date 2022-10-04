from sequences import kasami_seq, bin2sign, _norm_corr
from polys import PREFERRED
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from itertools import combinations
from scipy import stats


def main():

    deg = 16
    N = 2 ** (deg // 2)
    M = 5


    polys = PREFERRED[deg]
    kasami_codes = []
    kasami_ac = []
    kasami_psr = []

    for p in polys:

        # generate full set of kasami sequences and their autocorrelation
        for i in range(2, N - 1):
            kasami_code = bin2sign(kasami_seq(p, i))
            kasami_codes.append(kasami_code) 
            kasami_ac.append(_norm_corr(kasami_code, kasami_code))
            kasami_psr.append(psr_ratio(kasami_code))

        """# generate all cross-correlations
        gold_seq_pairs = combinations(gold_codes, 2)
        kasami_seq_pairs = combinations(kasami_codes, 2)
        for pair in gold_seq_pairs:
            gold_cc.append(np.correlate(pair[0], pair[1], mode='full'))

        for pair in kasami_seq_pairs:
            kasami_cc.append(np.correlate(pair[0], pair[1], mode='full'))"""

    # calculate peak to sidelobe ratio for all autocorrelations


    bind = np.argpartition(kasami_psr, -M)[-M:]

    best_kasami_psr = []
    best_kasami_psr_cc = []
    best_kasami_psr_ccoef = []

    for i in bind:
        best_kasami_psr.append(kasami_codes[i])
    
    for pair in combinations(best_kasami_psr, 2):
        cc_coef = stats.pearsonr(pair[0], pair[1])
        best_kasami_psr_cc.append(_norm_corr(pair[0], pair[1]))
        best_kasami_psr_ccoef.append(cc_coef.statistic)

    best_kasami_cc_ind = np.argmax(best_kasami_psr_ccoef)
    best_kasami_cc = best_kasami_psr_cc[best_kasami_cc_ind]


    figs_1 = make_subplots(rows=2, cols=1)

    figs_1.add_trace(go.Scatter(y=best_kasami_cc, mode='lines'), row=1, col=1)

    figs_1.add_trace(go.Scatter(y=best_kasami_psr_ccoef, mode='lines'), row=2, col=1)
    figs_1.show()


    
if __name__ == "__main__":
    main()
