from sequences import m_seq, gold_seq, kasami_seq, psr_ratio, acr_ratio, topk, _norm_corr
import numpy as np
from polys import GOLD
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import comb

from scipy import stats


def main():

    deg = 10
    mseq_nop = len(GOLD[deg][1])
    gold_nop = comb(len(GOLD[deg][1]), 2)

    mseq_pols = []
    mseq_psr = []
    mseq_acr = []
    mseq_ccf = []

    gold_pairs = []
    gold_psr = []
    gold_acr = []
    gold_ccf = []

    kasami_pairs = []
    kasami_psr = []
    kasami_acr = []
    kasami_ccf = []

    #11,12 11,17 300,10

    sel_a = 300
    sel_b = 10

    # generate full set of gold sequences and their autocorrelation and calculate peak to sidelobe ratio for all autocorrelations
    #for i in range(2, N - 1):

    ###################################
    # evaluation m-sequences
    for i in range(mseq_nop):
        mseq_code_a = m_seq(deg, sel_a, i)[0]
        mseq_code_b = m_seq(deg, sel_b, i)[0]
        cc_coef = stats.pearsonr(mseq_code_a, mseq_code_b)
        mseq_pols.append(str(m_seq(deg, sel_a, i)[1]))
        mseq_psr.append(0.5 * psr_ratio(mseq_code_a) + 0.5 * psr_ratio(mseq_code_b))
        mseq_acr.append(acr_ratio(mseq_code_a, mseq_code_b))
        mseq_ccf.append(cc_coef.statistic)

    bind = np.argmax(mseq_psr)
    mseq_best_psr_ac = _norm_corr(m_seq(deg, sel_a, bind)[0], m_seq(deg, sel_a, bind)[0])
    bind = np.argmax(mseq_acr)
    mseq_best_acr_cc = _norm_corr(m_seq(deg, sel_a, bind)[0], m_seq(deg, sel_b, bind)[0])

    ###################################
    # evaluation gold codes
    for i in range(gold_nop):
        gold_code_a = gold_seq(deg, sel_a, i)[0]
        gold_code_b = gold_seq(deg, sel_b, i)[0]
        cc_coef = stats.pearsonr(gold_code_a, gold_code_b)
        gold_pairs.append(str(gold_seq(deg, sel_a, i)[1]))
        gold_psr.append(0.5 * psr_ratio(gold_code_a) + 0.5 * psr_ratio(gold_code_b))
        gold_acr.append(acr_ratio(gold_code_a, gold_code_b))
        gold_ccf.append(cc_coef.statistic)

    bind = np.argmax(gold_psr)
    gold_best_psr_ac = _norm_corr(gold_seq(deg, sel_a, bind)[0], gold_seq(deg, sel_a, bind)[0])
    bind = np.argmax(gold_acr)
    gold_best_acr_cc = _norm_corr(gold_seq(deg, sel_a, bind)[0], gold_seq(deg, sel_b, bind)[0])

    ###################################
    #evaluation kasami codes
    kasami_code_a = kasami_seq(deg, sel_a)
    kasami_code_b = kasami_seq(deg, sel_b)
    cc_coef = stats.pearsonr(kasami_code_a, kasami_code_b)
    kasami_psr.append(0.5 * psr_ratio(kasami_code_a) + 0.5 * psr_ratio(kasami_code_b))
    kasami_acr.append(acr_ratio(kasami_code_a, kasami_code_b))
    kasami_ccf.append(cc_coef.statistic)

    bind = np.argmax(kasami_psr)
    kasami_best_psr_ac = _norm_corr(kasami_seq(deg, sel_a), kasami_seq(deg, sel_a))
    bind = np.argmax(kasami_acr)
    kasami_best_acr_cc = _norm_corr(kasami_seq(deg, sel_a), kasami_seq(deg, sel_b))

    figs_mseq = make_subplots(rows=2, cols=2)
    figs_mseq.add_trace(go.Scatter(x=mseq_pols, y=mseq_psr, mode='lines', name="PSR"), row=1, col=1)
    figs_mseq.add_trace(go.Scatter(x=mseq_pols, y=mseq_acr, mode='lines', name="ACR"), row=2, col=1)
    #figs_mseq.add_trace(go.Scatter(x=gold_pairs, y=gold_ccf, mode='lines', name="Pearson"), row=3, col=1)
    figs_mseq.add_trace(go.Scatter(y=mseq_best_psr_ac, mode='lines', name="best PSR AC"), row=1, col=2)
    figs_mseq.add_trace(go.Scatter(y=mseq_best_acr_cc, mode='lines', name="best ACR CC"), row=2, col=2)
    text = "Maximum length sequence code evaluation of degree " + str(deg)
    figs_mseq.update_layout(title_text = text)
    figs_mseq.show()


    figs_gold = make_subplots(rows=2, cols=2)
    figs_gold.add_trace(go.Scatter(x=gold_pairs, y=gold_psr, mode='lines', name="PSR"), row=1, col=1)
    figs_gold.add_trace(go.Scatter(x=gold_pairs, y=gold_acr, mode='lines', name="ACR"), row=2, col=1)
    #figs_gold.add_trace(go.Scatter(x=gold_pairs, y=gold_ccf, mode='lines', name="Pearson"), row=3, col=1)
    figs_gold.add_trace(go.Scatter(y=gold_best_psr_ac, mode='lines', name="best PSR AC"), row=1, col=2)
    figs_gold.add_trace(go.Scatter(y=gold_best_acr_cc, mode='lines', name="best ACR CC"), row=2, col=2)
    text = "Gold code evaluation of degree " + str(deg)
    figs_gold.update_layout(title_text = text)
    figs_gold.show()

    figs_kasami = make_subplots(rows=2, cols=2)
    figs_kasami.add_trace(go.Scatter(y=kasami_psr, mode='lines', name="PSR"), row=1, col=1)
    figs_kasami.add_trace(go.Scatter(y=kasami_acr, mode='lines', name="ACR"), row=2, col=1)
    #figs_kasami.add_trace(go.Scatter(y=kasami_ccf, mode='lines', name="Pearson"), row=3, col=1)
    figs_kasami.add_trace(go.Scatter(y=kasami_best_psr_ac, mode='lines', name="best PSR AC"), row=1, col=2)
    figs_kasami.add_trace(go.Scatter(y=kasami_best_acr_cc, mode='lines', name="best ACR CC"), row=2, col=2)
    text = "Kasami code evaluation of degree " + str(deg)
    figs_kasami.update_layout(title_text = text)
    figs_kasami.show()

    """# generate all cross-correlations
        gold_seq_pairs = combinations(gold_codes, 2)
        gold_seq_pairs = combinations(gold_codes, 2)
        for pair in gold_seq_pairs:
            gold_cc.append(np.correlate(pair[0], pair[1], mode='full'))

        for pair in gold_seq_pairs:
            gold_cc.append(np.correlate(pair[0], pair[1], mode='full'))"""

    """bind = topk(gold_psr, M)
    wind = topk(gold_psr, -M)


    best_gold_psr = []
    best_gold_psr_cc = []
    best_gold_psr_ac = []
    best_gold_psr_ccoef = []

    for i in bind:
        best_gold_psr.append(gold_codes[i])

    figs_bpsr = make_subplots(rows=10, cols=1)
    j = 1
    for i in bind[:9]:
        ac = _norm_corr(gold_codes[i], gold_codes[i])
        figs_bpsr.add_trace(go.Scatter(y=ac, mode='lines', name=str(i)), row=j, col=1)
        j += 1
    figs_bpsr.update_layout(title_text = "Top 10 PSR Autocorrelations")
    figs_bpsr.show()

    figs_wpsr = make_subplots(rows=10, cols=1)
    j = 1
    for i in wind[:9]:
        ac = _norm_corr(gold_codes[i], gold_codes[i])
        figs_wpsr.add_trace(go.Scatter(y=ac, mode='lines', name=str(i)), row=j, col=1)
        j += 1
    figs_wpsr.update_layout(title_text = "Worst 10 PSR Autocorrelations")
    figs_wpsr.show()
    
    for pair in combinations(best_gold_psr, 2):
        cc_coef = stats.pearsonr(pair[0], pair[1])
        best_gold_psr_cc.append(_norm_corr(pair[0], pair[1]))
        best_gold_psr_ac.append([_norm_corr(pair[0], pair[0]),_norm_corr(pair[1], pair[1])])
        best_gold_psr_ccoef.append(cc_coef.statistic)

    best_gold_cc_ind = np.argmax(best_gold_psr_ccoef)
    best_gold_psr_ac = best_gold_psr_ac[best_gold_cc_ind]
    best_gold_cc = best_gold_psr_cc[best_gold_cc_ind]


    figs_1 = make_subplots(rows=3, cols=1)

    for ac in best_gold_psr_ac:
        figs_1.add_trace(go.Scatter(y=ac, mode='lines', name='Autocorrelation'), row=1, col=1)

    figs_1.add_trace(go.Scatter(y=best_gold_cc, mode='lines', name='Cross-correlation'), row=2, col=1)

    figs_1.add_trace(go.Scatter(y=best_gold_psr_ccoef, mode='lines', name='Cross-correlation coefficent'), row=3, col=1)
    figs_1.show()"""


    
if __name__ == "__main__":
    main()
