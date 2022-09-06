from polys import PREFERRED
from scipy import signal as sig
import numpy as np
import plotly.express as px

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

# @TODO: fix kasami, implement all steps until waveform export 

def main():

    deg = 9

    # select preferred polynomials
    poly_a = PREFERRED[deg][0]
    poly_b = PREFERRED[deg][1]

    # convert octal polynomials to binary array
    poly_a = oct2binarray(poly_a)
    poly_b = oct2binarray(poly_b)

    print(poly_a)
    print(poly_b)

    # generate maximal length sequence from polynomials
    code_a = sig.max_len_seq(deg + 1, poly_a)[0]
    code_b = sig.max_len_seq(deg + 1, poly_b)[0]

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