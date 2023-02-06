from polys import GOLD, KASAMI
from scipy import signal as sig
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations

def _bin2sign(code: np.ndarray):
    """Replaces binary with signed values

    Replaces values 0 and 1 with +1 and -1.

    Args
    ----
    code: np.ndarray
        code as binary array

    Returns
    -------
    signed values as array
    """
    code[code == 1] = -1
    code[code == 0] = 1
    return code

def _oct2poly(code_oct: int):
    """Converts octal polynom representation to polynomial degrees array

    Gets octal representation as string to converts it into a integer.
    It then converts it to a binary representation and iterates through every binary number to append it as an integer array.
    After that it calcualtes the degrees of the binary polynom an appends them to an array.
    
    Args
    ----
    code_oct: int
        octal number in integer format

    Returns
    -------
    array of integer degrees of polynomial
    """
    code = [int(d) for d in bin(int(str(code_oct), 8))[2:]]
    poly = []
    for i in range(len(code)):
        if code[i]:
            poly.append(code[i] * (len(code) - (i + 1)))
    return poly

def _norm_corr(data_x: list[int], data_y: list[int]):
    corr = sig.correlate(data_x, data_y, 'full')
    return corr / len(data_x)

def topk(data: list[int], k: int):
    ind = []
    data = np.asarray(data)
    ind = np.argpartition(data, -k)[-k:]
    ind[np.argsort(data[ind])]
    return ind

def psr_ratio(data: list[int]):
    ac = sig.correlate(data, data, 'full')
    ratio = (np.max(ac) - np.mean(ac)) / np.std(ac)
    return ratio

def acr_ratio(data_a: list[int], data_b: list[int]):
    ac = 0.5 * sig.correlate(data_a, data_a, 'full') + 0.5 * sig.correlate(data_b, data_b, 'full')
    cc = sig.correlate(data_a, data_b, 'full')
    ratio = np.max(ac) / np.max(cc)
    return ratio

def m_seq(deg: int, ind: int = 0, pind: int = 1):
    poly = GOLD[deg][1][pind]
    poly_u = _oct2poly(poly)
    init = (deg - 1) * [0] + [1]
    seq_u = sig.max_len_seq(deg, init, taps=poly_u)[0]
    seq_u = np.roll(seq_u, 1)
    seq_u = np.roll(seq_u, -ind)
    return _bin2sign(seq_u.astype('float64')), poly

def gold_seq(deg: int, ind: int = 3, comb: int = 1):
    """Generates gold sequences

    Needs 2 maximum length sequences and uses them to make gold sequences by circular shifting and XORing.
    There are 2^n + 1 different gold sequences generatable by shifting, n is the degree of the generator polynomial.

    Args
    ----
    seq_u: np.ndarray
        numpy array of frist m-sequence
    seq_v: np.ndarray
        numpy array of second m-sequence
    index: int
        indexation of gold sequence set
    
    Returns
    -------
    numpy array of gold sequence
    """
    polys = list(combinations(GOLD[deg][1], 2))[comb]
    poly_u = _oct2poly(polys[0])
    poly_v = _oct2poly(polys[1])
    init = (deg - 1) * [0] + [1]
    seq_u = sig.max_len_seq(deg, init, taps=poly_u)[0]
    seq_v = sig.max_len_seq(deg, init, taps=poly_v)[0]
    seq_u = np.roll(seq_u, 1)
    seq_v = np.roll(seq_v, 1)

    if ind == 0:
        return _bin2sign(seq_u.astype('float64')), polys
    elif ind == 1:
        return _bin2sign(seq_v.astype('float64')), polys
    else:
        seq_v = np.roll(seq_v, 2 - ind)
        code = seq_u ^ seq_v
        return _bin2sign(code.astype('float64')), polys

def kasami_seq(deg: int, ind: int = 3):
    """Generates kasami sequences

    Needs one maximum length sequence and uses it to generate (small set of) kasami sequences by decimation, circular shiting and XORing.
    There are 2^(n/2) different kasami sequences generatable by shifting, n is the degree of the generator polynomial.

    Args
    ----
    seq_u: np.ndarray
        numpy array of m-sequence
    index: int
        indexation of kasami sequence set

    Returns
    -------
    numpy array of kasami sequence
    """
    poly_u = _oct2poly(KASAMI[deg][0])
    poly_v = _oct2poly(KASAMI[deg][1])
    init = (deg - 1) * [0] + [1]
    seq_u = sig.max_len_seq(deg, init, taps=poly_u)[0]
    seq_v = sig.max_len_seq(deg, init, taps=poly_v)[0]
    seq_u = np.roll(seq_u, 1)
    seq_v = np.roll(seq_v, 1)

    if ind == 0:
        return _bin2sign(seq_u.astype('float64'))
    elif ind == 1:
        return _bin2sign(seq_v.astype('float64'))
    else:
        seq_v = np.roll(seq_v, 2 - ind)
        code = seq_u ^ seq_v
        return _bin2sign(code.astype('float64'))

# @TODO: fix kasami, implement all steps until waveform export, basisband 

def main():

    deg = 8

    gold_set_size = 2 ** deg + 1
    kasami_set_size = 2 ** (deg // 2)

    gold_codes = []
    kasami_codes = []

    gold_ac = []
    kasami_ac = []
    gold_cc = []
    kasami_cc = []
    gold_psr = []
    gold_acr = []
    kasami_psr = []
    kasami_acr = []

    # select preferred polynomials
   # polys = PREFERRED[deg]


    # generate full set of gold sequences and their autocorrelation
    for i in range(2, gold_set_size):
        gold_code = _bin2sign(gold_seq(polys[0], polys[1], i))
        gold_codes.append(gold_code)
        gold_ac.append(_norm_corr(gold_code, gold_code))

    # generate full set of kasami sequences and their autocorrelation
    for i in range(2, kasami_set_size):
        kasami_code = _bin2sign(kasami_seq(polys[0], i))
        kasami_codes.append(kasami_code) 
        kasami_ac.append(_norm_corr(kasami_code, kasami_code))

    # generate all cross-correlations
    gold_seq_pairs = combinations(gold_codes, 2)
    kasami_seq_pairs = combinations(kasami_codes, 2)
    for pair in gold_seq_pairs:
        gold_cc.append(np.correlate(pair[0], pair[1], mode='full'))

    for pair in kasami_seq_pairs:
        kasami_cc.append(np.correlate(pair[0], pair[1], mode='full'))

    # calculate peak to sidelobe ratio for all autocorrelations
    for ac in gold_ac:
        gold_psr.append((np.max(ac) - np.mean(ac)) / np.std(ac))

    for ac in kasami_ac:
        kasami_psr.append((np.max(ac) - np.mean(ac)) / np.std(ac))


    best_gold_psr = np.argmax(gold_psr)
    worst_gold_psr = np.argmin(gold_psr)
    best_kasami_psr = np.argmax(kasami_psr)
    worst_kasami_psr = np.argmin(kasami_psr)

    figs_1 = make_subplots(rows=1, cols=2)
    #figs_1.add_trace(go.Scatter(y=(gold_ac[best_gold_psr]), mode='lines', name='Best Gold Autocorrelation'), row=1, col=1)
    figs_1.add_trace(go.Scatter(y=(kasami_ac[best_kasami_psr]), mode='lines', name='Best Kasami Autocorrelation'), row=1, col=1)
    #figs_1.add_trace(go.Scatter(y=(gold_ac[worst_gold_psr]), mode='lines', name='Worst Gold Autocorrelation'), row=1, col=2)
    figs_1.add_trace(go.Scatter(y=(kasami_ac[worst_kasami_psr]), mode='lines', name='Worst Kasami Autocorrelation'), row=1, col=2)

    figs_2 = make_subplots(rows=2, cols=1)
    figs_2.add_trace(go.Scatter(y=gold_psr, mode='lines', name='Gold set PSR'), row=1, col=1)
    figs_2.add_trace(go.Scatter(y=kasami_psr, mode='lines', name='Kasami set PSR'), row=2, col=1)

    figs_1.show()
    figs_2.show()

    fig = px.line(gold_acr)
    fig.show()

    fig = px.line(kasami_acr)
    fig.show()


if __name__ == "__main__":
    #main()
    pass
