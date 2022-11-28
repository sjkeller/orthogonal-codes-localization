from sequences import m_seq, gold_seq, kasami_seq, psr_ratio, acr_ratio, topk, _norm_corr
import numpy as np
from polys import GOLD
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import comb

from scipy import stats

DPIEXPORT = 400

def main():

    deg = [6,7,8,9,10, 11]

    kasami_psr_means = []
    gold_psr_means = []
    mseq_psr_means = []

    kasami_acr_means = []
    gold_acr_means = []
    mseq_acr_means = []

    for d in deg:
        mseq_nop = len(GOLD[d][1])
        gold_nop = comb(len(GOLD[d][1]), 2)

        mseq_pols = []
        mseq_psr = []
        mseq_acr = []
        mseq_ccf = []

        gold_pairs = []
        gold_psr = []
        gold_acr = []
        gold_ccf = []

        kasami_psr = []
        kasami_acr = []
        kasami_ccf = []

        #11,12 11,17 300,10
        N = 1000
        # sel_a = 300
        # sel_b = 10

        sel_a = np.random.rand(N) * 150
        sel_b = np.random.rand(N) * 150

        print(sel_a, sel_b)

        sel_a = [int(i) for i in sel_a]
        sel_b = [int(i) for i in sel_b]

        # generate full set of gold sequences and their autocorrelation and calculate peak to sidelobe ratio for all autocorrelations
        #for i in range(2, N - 1):

        ###################################
        # evaluation m-sequences

        norm_psr = []
        norm_acr = []

        for i in range(mseq_nop):

            for sa, sb in zip(sel_a, sel_b):

                mseq_code_a = m_seq(d, sa, i)[0]
                mseq_code_b = m_seq(d, sb, i)[0]
                cc_coef = stats.pearsonr(mseq_code_a, mseq_code_b)

                norm_psr.append(0.5 * psr_ratio(mseq_code_a) + 0.5 * psr_ratio(mseq_code_b))
                norm_acr.append(acr_ratio(mseq_code_a, mseq_code_b))


            mseq_pols.append(str(m_seq(d, sa, i)[1]))
            mseq_psr.append(np.mean(norm_psr))
            mseq_acr.append(np.mean(norm_acr))


            mseq_ccf.append(cc_coef.statistic)

        bind = np.argmax(mseq_psr)
        mseq_best_psr_ac = _norm_corr(m_seq(d, sa, bind)[0], m_seq(d, sa, bind)[0])
        bind = np.argmax(mseq_acr)
        mseq_best_acr_cc = _norm_corr(m_seq(d, sa, bind)[0], m_seq(d, sb, bind)[0])

        ###################################
        # evaluation gold codes

        norm_psr = []
        norm_acr = []
        for i in range(gold_nop):

            for sa, sb in zip(sel_a, sel_b):
                gold_code_a = gold_seq(d, sa, i)[0]
                gold_code_b = gold_seq(d, sb, i)[0]
                cc_coef = stats.pearsonr(gold_code_a, gold_code_b)
                #gold_pairs.append(str(gold_seq(d, sel_a, i)[1]))

                norm_psr.append(0.5 * psr_ratio(gold_code_a) + 0.5 * psr_ratio(gold_code_b))
                norm_acr.append(acr_ratio(gold_code_a, gold_code_b))

            gold_psr.append(np.mean(norm_psr))
            gold_acr.append(np.mean(norm_acr))
            gold_ccf.append(cc_coef.statistic)

        bind = np.argmax(gold_psr)
        gold_best_psr_ac = _norm_corr(gold_seq(d, sa, bind)[0], gold_seq(d, sa, bind)[0])
        bind = np.argmax(gold_acr)
        gold_best_acr_cc = _norm_corr(gold_seq(d, sa, bind)[0], gold_seq(d, sb, bind)[0])

        ###################################
        #evaluation kasami codes
        norm_psr = []
        norm_acr = []        
        
        if not d % 2:

            for sa, sb in zip(sel_a, sel_b):
                kasami_code_a = kasami_seq(d, sa)
                kasami_code_b = kasami_seq(d, sb)
                cc_coef = stats.pearsonr(kasami_code_a, kasami_code_b)

                norm_psr.append(0.5 * psr_ratio(kasami_code_a) + 0.5 * psr_ratio(kasami_code_b))
                norm_acr.append(acr_ratio(kasami_code_a, kasami_code_b))


            kasami_psr.append(np.mean(norm_psr))
            kasami_acr.append(np.mean(norm_acr))


            kasami_ccf.append(cc_coef.statistic)

            bind = np.argmax(kasami_psr)
            kasami_best_psr_ac = _norm_corr(kasami_seq(d, sa), kasami_seq(d, sa))
            bind = np.argmax(kasami_acr)
            kasami_best_acr_cc = _norm_corr(kasami_seq(d, sa), kasami_seq(d, sb))

 

        kasami_acr_means.append(np.mean(kasami_acr))
        gold_acr_means.append(np.mean(gold_acr))
        mseq_acr_means.append(np.mean(mseq_acr))

        kasami_psr_means.append(np.mean(kasami_psr))
        gold_psr_means.append(np.mean(gold_psr))
        mseq_psr_means.append(np.mean(mseq_psr))


    acr_means_max = np.nanmax(kasami_acr_means + gold_acr_means + mseq_acr_means)
    psr_means_max = np.nanmax(kasami_psr_means + gold_psr_means + mseq_psr_means)

    kasami_acr_means /= acr_means_max
    gold_acr_means /= acr_means_max
    mseq_acr_means /= acr_means_max

    kasami_psr_means /= psr_means_max
    gold_psr_means /= psr_means_max
    mseq_psr_means /= psr_means_max

    print("ACR:", kasami_acr_means, gold_acr_means, mseq_acr_means)

    print("PSR:", kasami_psr_means, gold_psr_means, mseq_psr_means)

    fig_eva = go.Figure()
    fig_eva.add_trace(go.Bar(x=deg, y=mseq_acr_means, name='m-seq', marker_color='#1F77B4'))
    fig_eva.add_trace(go.Bar(x=deg, y=gold_acr_means, name='Gold', marker_color='#F58518'))
    fig_eva.add_trace(go.Bar(x=deg, y=kasami_acr_means, name='Kasami', marker_color='#00CC96'))

    fig_eva.update_layout(xaxis_title='degree', yaxis_title='relative ACR')

    fig_eva.show()
    fig_eva.write_image("img/degAcrEva.pdf", scale=1, width= 1.6 * DPIEXPORT, height= 1.2 * DPIEXPORT)

    del fig_eva

    fig_eva = go.Figure()
    fig_eva.add_trace(go.Bar(x=deg, y=mseq_psr_means, name='m-seq', marker_color='#1F77B4'))
    fig_eva.add_trace(go.Bar(x=deg, y=gold_psr_means, name='Gold', marker_color='#F58518'))
    fig_eva.add_trace(go.Bar(x=deg, y=kasami_psr_means, name='Kasami', marker_color='#00CC96'))

    fig_eva.update_layout(xaxis_title='degree', yaxis_title='relative PSR')

    fig_eva.show()
    fig_eva.write_image("img/degPsrEva.pdf", scale=1, width= 1.6 * DPIEXPORT, height= 1.2 * DPIEXPORT)

    del fig_eva


    """figs_mseq = make_subplots(rows=1, cols=2, subplot_titles=('PSR per polynomial [oct]', 'autocorrelation of max PSR'))
    figs_mseq.add_trace(go.Scatter(x=mseq_pols, y=mseq_psr, mode='lines', name="PSR", marker_color='#000'), row=1, col=1)
    figs_mseq.add_trace(go.Scatter(y=mseq_best_psr_ac, mode='lines', name="best PSR AC", marker_color='#000'), row=1, col=2)
    figs_mseq.update_layout(showlegend=False)
    #figs_mseq.show()
    #figs_mseq.write_image("img/mseq_eva_psr.pdf", scale=1, width= 2.5 * DPIEXPORT, height= 1 * DPIEXPORT)


    figs_mseq = make_subplots(rows=1, cols=2, subplot_titles=('ACR per polynomial [oct]', 'cross-correlation of max ACR'))
    #figs_mseq.add_trace(go.Scatter(x=gold_pairs, y=gold_ccf, mode='lines', name="Pearson"), row=3, col=1)
    figs_mseq.add_trace(go.Scatter(x=mseq_pols, y=mseq_acr, mode='lines', name="ACR", marker_color='#000'), row=1, col=1)
    figs_mseq.add_trace(go.Scatter(y=mseq_best_acr_cc, mode='lines', name="best ACR CC", marker_color='#000'), row=1, col=2)
    figs_mseq.update_layout(showlegend=False)
    #figs_mseq.show()
    #figs_mseq.write_image("img/mseq_eva_acr.pdf", scale=1, width= 2.5 * DPIEXPORT, height= 1 * DPIEXPORT)


    figs_gold = make_subplots(rows=1, cols=2, subplot_titles=('PSR per polynomial [oct]', 'autocorrelation of max PSR'))
    figs_gold.add_trace(go.Scatter(x=gold_pairs, y=gold_psr, mode='lines', name="PSR", marker_color='#000'), row=1, col=1)
    figs_gold.add_trace(go.Scatter(y=gold_best_psr_ac, mode='lines', name="best PSR AC", marker_color='#000'), row=1, col=2)
    figs_gold.update_layout(showlegend=False)
    #figs_gold.show()
    #figs_gold.write_image("img/gold_eva_psr.pdf", scale=1, width= 2.5 * DPIEXPORT, height= 1 * DPIEXPORT)
    figs_gold = make_subplots(rows=1, cols=2, subplot_titles=('ACR per polynomial [oct]', 'cross-correlation of max ACR'))
    #figs_gold.add_trace(go.Scatter(x=gold_pairs, y=gold_ccf, mode='lines', name="Pearson"), row=3, col=1)
    figs_gold.add_trace(go.Scatter(x=gold_pairs, y=gold_acr, mode='lines', name="ACR", marker_color='#000'), row=1, col=1)
    figs_gold.add_trace(go.Scatter(y=gold_best_acr_cc, mode='lines', name="best ACR CC", marker_color='#000'), row=1, col=2)
    figs_gold.update_layout(showlegend=False)
    #figs_gold.show()
    #figs_gold.write_image("img/gold_eva_acr.pdf", scale=1, width= 2.5 * DPIEXPORT, height= 1 * DPIEXPORT)

    figs_kasami = make_subplots(rows=2, cols=2)
    figs_kasami.add_trace(go.Scatter(y=kasami_psr, mode='lines', name="PSR", marker_color='#EF553B'), row=1, col=1)
    figs_kasami.add_trace(go.Scatter(y=kasami_acr, mode='lines', name="ACR", marker_color='#636EFA'), row=2, col=1)
    #figs_kasami.add_trace(go.Scatter(y=kasami_ccf, mode='lines', name="Pearson"), row=3, col=1)
    figs_kasami.add_trace(go.Scatter(y=kasami_best_psr_ac, mode='lines', name="best PSR AC", marker_color='#EF553B'), row=1, col=2)
    figs_kasami.add_trace(go.Scatter(y=kasami_best_acr_cc, mode='lines', name="best ACR CC", marker_color='#636EFA'), row=2, col=2)
    text = "Kasami code evaluation of degree " + str(deg)
    figs_kasami.update_layout(title_text = text)
    #figs_kasami.show()

    scFa = 1

    #figs_mseq.write_image("img/mseq_eva.pdf", scale=scFa, width= 1.5 * DPIEXPORT, height= 2 * DPIEXPORT)
    #figs_gold.write_image("img/gold_eva.pdf", scale=scFa, width= 1.5 * DPIEXPORT, height= 2 * DPIEXPORT)
    #figs_kasami.write_image("img/kasami_eva.pdf", scale=scFa, width= 1.5 * DPIEXPORT, height= 2 * DPIEXPORT)

    # generate all cross-correlations
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
    
