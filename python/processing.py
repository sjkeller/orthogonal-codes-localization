from re import T
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from sequences import gold_seq
from polys import GOLD

from commpy import filters as flt
from scipy import signal as sig
from scipy.io import savemat

### gold sequence generation
deg = 6
gold_nop = len(GOLD[deg][1])
rawCode = gold_seq(deg, 1, 1)[0]

### signal parameters
fs = 200e3          # sampling rate
bw = 20e3           # bandwith
Tsym = 1 / bw       # symbol length
SpS = fs * Tsym     # upsampling factor
rolloff = 1/8       # FIR cosine filter coefficent
fc = 62.5e3         # carrier freqency

### upsample and filter via FIR cosine
filter = flt.rcosfilter(512, rolloff, Tsym, fs)[1]
tSigBB = sig.resample_poly(rawCode, SpS, 1, window=filter)
time = np.linspace(0, Tsym * len(rawCode), len(tSigBB))

fSigBB = np.fft.fft(tSigBB)
fBB = np.fft.fftfreq(fSigBB.shape[0], 1/fs)

### shift spectrum of SigBB to carrier frequency
tSigTB = np.real(tSigBB * np.exp(2.j*np.pi*fc*time))

fSigTB = np.fft.fft(tSigTB)
fTB = np.fft.fftfreq(fSigTB.shape[0], 1/fs)

### shifting zero-frequency origin to the center
fSigBB = np.fft.fftshift(fSigBB)
fBB = np.fft.fftshift(fBB)
fSigTB = np.fft.fftshift(fSigTB)
fTB = np.fft.fftshift(fTB)

figure = make_subplots(rows=3, cols=1)
figure.add_trace(go.Bar(y=rawCode, name="bit domain"), row=1, col=1)
figure.add_trace(go.Scatter(x=time, y=tSigBB, mode='lines', name="baseband signal"), row=2, col=1)
figure.add_trace(go.Scatter(x=time, y=tSigTB, mode='lines', name="carrier signal"), row=2, col=1)
figure.add_trace(go.Scatter(x=fBB, y=np.abs(fSigBB), fill='tozeroy', mode='lines', name="baseband"), row=3, col=1)
figure.add_trace(go.Scatter(x=fTB, y=np.abs(fSigTB), fill='tozeroy', mode='lines', name="carrier"), row=3, col=1)
figure.show()

matobj = {'tSigTB': tSigTB}
savemat('/Users/sk/Library/CloudStorage/OneDrive-Persönlich/Studium/TUHH/3. Semester Master/Forschungsprojekt/keller_orthogonal-codes/python/output/signal.mat', matobj)

### processing backwards

tSigBBr = np.real(tSigTB * np.exp(2.j*np.pi*-fc*time))
fSigBBr = np.fft.fft(tSigBBr)
fBBr = np.fft.fftfreq(fSigBBr.shape[0], 1/fs)

fSigBBr = np.fft.fftshift(fSigBBr)
fBBr = np.fft.fftshift(fBBr)
# low pass filter after backward shifting


figure = make_subplots(rows=3, cols=1)
figure.add_trace(go.Scatter(x=fBBr, y=np.abs(fSigBBr), mode='lines', name="baseband signal"), row=2, col=1)
figure.show()