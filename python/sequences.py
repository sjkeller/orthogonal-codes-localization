from scipy import signal as sig
import numpy as np
import plotly.express as px

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

test = sig.max_len_seq(8, [0, 0, 0, 0, 0, 0, 0, 1])[0]

print(type(test))
fig = px.bar(test)
fig.show()

code_a = sig.max_len_seq(8, [1, 0, 0, 1, 0, 0, 1, 1])[0]
code_b = sig.max_len_seq(8, [1, 1, 0, 0, 1, 0, 0, 1])[0]

gold_code = gold_seq(code_a, code_b, 4)
fig = px.bar(gold_code)
fig.show()

kasami_code = kasami_seq(code_a, 4, 8)
fig = px.bar(kasami_code)
fig.show()

# @TODO: fix kasami, implement all steps until waveform export 