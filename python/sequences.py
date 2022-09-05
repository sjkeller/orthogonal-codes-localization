from polys import PREFERRED
from scipy import signal as sig
import numpy as np
import plotly.express as px

def oct2binarray(code_oct: int):
    code = [int(d) for d in bin(int(str(code_oct), 8))[2:]]
    return code

def gold_seq(seq_u: np.ndarray, seq_v: np.ndarray, shift: int):
    seq_v = np.roll(seq_v, shift)
    return seq_u ^ seq_v

def kasami_seq(seq_u: np.ndarray, shift: int, deg: int):
    seq_w = seq_u
    dec = 1 + 2 ** (deg // 2) 
    print(dec)
    seq = seq_w[::dec]
    seq[:] = 0
    seq_w = np.roll(seq_w, shift)
    return seq_u ^ seq_w

# @TODO: fix kasami, implement all steps until waveform export 

def main():

    deg = 9

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

    # generate gold codes
    gold_code = gold_seq(code_a, code_b, 4)

    # generate kasami codes
    kasami_code = kasami_seq(code_a, 4, deg)

    fig = px.bar(gold_code)
    fig.show()

    fig = px.bar(kasami_code)
    fig.show()

if __name__ == "__main__":
    main()