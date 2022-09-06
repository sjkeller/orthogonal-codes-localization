from polys import PREFERRED
from scipy import signal as sig
from itertools import combinations
import numpy as np
import plotly.express as px

def bin2sign(code: np.ndarray):
    """Replaces binary with signed values

    Replaces values 0 and 1 with +1 and -1.

    Args
    ----
    code: np.ndarray
        code as binary array

    Returns
    -------
    - signed values as array
    """
    code[code == 1] = -1
    code[code == 0] = 1
    return code

def oct2binarray(code_oct: int):
    """Converts octal to binary arrays

    Gets octal representation as string to convet it into a integer.
    It then converts it to a binary representation and iterates through every binary number to append it as an integer array.
    
    Args
    ----
    code_oct: int
        octal number in integer format

    Returns
    -------
    - array of binaries
    """
    code = [int(d) for d in bin(int(str(code_oct), 8))[2:]]
    return code

def gold_seq(seq_u: np.ndarray, seq_v: np.ndarray, shift: int):
    """Generates gold sequences

    Needs 2 maximum length sequences and uses them to make gold sequences by circular shifting and XORing.
    There are 2^n + 1 different gold sequences generatable by shifting, n is the degree of the generator polynomial.

    Args
    ----
    seq_u: np.ndarray
        numpy array of frist m-sequence
    seq_v: np.ndarray
        numpy array of second m-sequence
    shift: int
        shift parameter of gold sequence generation. 
    
    Returns
    -------
    - numpy array of gold sequence

    """
    seq_v = np.roll(seq_v, shift)
    return seq_u ^ seq_v

def kasami_seq(seq_u: np.ndarray, shift: int, deg: int):
    """Generates kasami sequences

    Needs one maximum length sequence and uses it to generate (small set of) kasami sequences by decimation, circular shiting and XORing.
    There are 2^(n/2) different kasami sequences generatable by shifting, n is the degree of the generator polynomial.

    Args
    ----
    seq_u: np.ndarray
        numpy array of m-sequence
    shift: int
        shift parameter of kasami sequence generation

    Returns
    -------
    - numpy array of kasami sequence
    """
    seq_w = seq_u
    dec = 1 + 2 ** (deg // 2) 
    seq = seq_w[::dec]
    seq[:] = 0

    seq_w = np.roll(seq_w, shift)
    return seq_u ^ seq_w

# @TODO: fix kasami, implement all steps until waveform export, basisband 

def main():

    deg = 10

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
    poly_a = PREFERRED[deg - 1][0]
    poly_b = PREFERRED[deg - 1][1]

    # convert octal polynomials to binary array
    poly_a = oct2binarray(poly_a)
    poly_b = oct2binarray(poly_b)

    print(poly_a)
    print(poly_b)

    # generate maximal length sequence from polynomials
    code_a = sig.max_len_seq(deg, poly_a)[0]
    code_b = sig.max_len_seq(deg, poly_b)[0]

    # generate full set of gold sequences and their autocorrelation
    for i in range(gold_set_size):
        gold_code = bin2sign(gold_seq(code_a, code_b, i))
        gold_codes.append(gold_code)
        gold_ac.append(np.correlate(gold_code, gold_code, mode='full'))
        """if i == 800:
            fig = px.line(gold_ac[-1])
            fig.show()"""


    # generate full set of kasami sequences and their autocorrelation
    for i in range(kasami_set_size):
        kasami_code = bin2sign(kasami_seq(code_a, i, deg))
        kasami_codes.append(kasami_code) 
        kasami_ac.append(np.correlate(kasami_code, kasami_code, mode='full'))


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



    fig = px.line(gold_psr)
    fig.show()

    fig = px.line(kasami_psr)
    fig.show()

    """fig = px.line(gold_acr)
    fig.show()

    fig = px.line(kasami_psr)
    fig.show()

    fig = px.line(kasami_acr)
    fig.show()"""

if __name__ == "__main__":
    main()
