import numpy as np
import plotly.graph_objects as go
from commpy import filters as flt
from plotly.subplots import make_subplots
from scipy import signal as sig
from scipy.io import savemat, loadmat, wavfile
from sequences import gold_seq

import matplotlib.pyplot as plt

DPIEXPORT = 400
MATSAVE = '/Users/sk/Library/CloudStorage/OneDrive-Persönlich/Studium/TUHH/3. Semester Master/Forschungsprojekt/uw-watermark-main/Watermark/input/signals/tSigTB_'
WAVLOAD = '/Users/sk/Library/CloudStorage/OneDrive-Persönlich/Studium/TUHH/3. Semester Master/Forschungsprojekt/uw-watermark-main/Watermark/output/'

CASTLE_1_SELECT = [0, 0, 0]

file_index = 0
load_index = 0

### signal parameters
fs = 200e3          # sampling rate
bw = 20e3           # bandwith
Tsym = 1 / bw       # symbol length
SpS = fs * Tsym     # upsampling factor
rolloff = 1/8       # FIR cosine filter coefficent
fc = 62.5e3         # carrier freqency
sysOrd = 3          # order of butterworth filter

#targetSNR = 0.1        # targeted Signal Noise Ratio for addtive GWN generator
watermarkDelay = 12e-3  # delay of watermark simulation time of flight

### gold sequence generation
#deg = 10


def genAxis(size: int, step: float):
    newaxis = np.zeros(size)
    i = 1
    while i < size:
        newaxis[i] = newaxis[i - 1] + step
        i += 1
    return newaxis

def testPlotting(sig, time, num):
    fig = make_subplots(rows=2, cols=1)
    print(len(sig))
    fig.add_trace(go.Scatter(y=np.real(sig), x=time), row=1, col=1)
    fSigT, freqT = _get_fftfunc(sig, fs)
    fig.add_trace(go.Scatter(y=freqT, x=fSigT), row=2, col=1)
    mytitle = "Test " + str(num)
    fig.update_layout(title=mytitle)
    fig.show()


def _get_fftfunc(sig: np.ndarray, fs: float):
    """applies central shifted fast foruiert transformation

    Args
    ----
    sig: np.ndarray
        signal
    fs: float
        sample rate

    Returns
    -------
    fast fourier transform and its frequencies
    """

    fSig = np.fft.fft(sig)
    freq = np.fft.fftfreq(fSig.shape[0], 1/fs)
    fSig = np.fft.fftshift(fSig)
    freq = np.fft.fftshift(freq)

    return (freq, np.abs(fSig))

def gen_tb_signal(time: np.ndarray, tSigBB: np.ndarray, saveMat: bool = True, polyDeg: int = 10):
    """Generates transfer band signal and saves its optionally as mat file

    Args
    ----
    time: np.ndarray
        time axis of signal
    tSigBB: np.ndarray
        baseband signal
    saveMat:
        saves or overwrites mat export for watermark

    Returns
    -------
    time axis and transferband shifted signal
    """

    

    ### shift spectrum to transmission band
    tSigTB = tSigBB * np.exp(2.j*np.pi*fc*time)

    ### save signal as matlab object for watermark
    matobj = {'fs_x': fs, 'nBits': 0, 'x': tSigTB}
    if saveMat:
        global file_index
        savemat(MATSAVE + "d" + str(polyDeg) + "_" + str(file_index) + '.mat', matobj)
        file_index += 1

    return time, tSigTB

def get_tb_signal(sigLen: int, channel: str, polyDeg: int = 10):
    """ Shifts transfer band signal back to baseband

    Args
    ----
    time: np.ndrarray
        time axis of signal
    tSigTB: np.ndarray
        trasnfer band signal

    Returns
    -------
    time axis and baseband signal
    """

    global load_index

    filepath = channel + "/tSigTB_" + "d" + str(polyDeg) + "_" + str(load_index) + "/" + channel + "_001.wav"

    _, tSigTBr = wavfile.read(WAVLOAD + filepath)

    waveLen = loadmat(WAVLOAD + channel + "/tSigTB_" + "d" + str(polyDeg) + "_" + str(load_index) + "/bookkeeping.mat")

    #waveLen = waveLen['bk'][0][0][2][0][1]
    lstWaveInd = waveLen['bk'][0][0][2]
    sample_pick = np.random.randint(0, len(lstWaveInd))
    #sample_pick = CASTLE_1_SELECT[load_index]
    #print("index", lstWaveInd[sample_pick])
    #tSigTBr = tSigTBr[(lstWaveInd[sample_pick][0] + int(watermarkDelay * fs)):(lstWaveInd[sample_pick][0] + int(sigLen + watermarkDelay * fs))]
    tSigTBr = tSigTBr[(lstWaveInd[sample_pick][0] + int(watermarkDelay * fs)):(lstWaveInd[sample_pick][1])]
    #time = np.linspace(0, len(tSigTBr) * Tsym, len(tSigTBr))
    load_index += 1


    return tSigTBr

def delay_sum(signals: list[np.ndarray], delays: list[float], fs: float, stepping: float):
    """Creates a delayed sum of signals

    Padds every signal with its custom delay and extends their length to a size where nothing cuts off.
    At the end all delayed and padded signals get summed up.

    Args
    ----
    signals: list[np.ndarray]
        list of baseband signals
    delays: list[float]
        list of delays to be applied
    fs: float
        sample rate
    watmark: bool
        takes watermak 12ms delay into account for delay padding

    Returns
    -------
    time axis of whole sum and the sum of signals
    """

    singalSumLen = int(np.floor(np.max(delays) * fs) + len(signals[0]) + 1)
    signalSum = np.zeros(singalSumLen, dtype = 'complex_')
    timeSum = genAxis(singalSumLen, stepping)
    
    for s, d in zip(signals, delays):

        index = range(int(np.floor(d * fs) + 1), int((np.floor(d * fs) + len(s) + 1)))
        signalSum[index] = signalSum[index] + s

    return timeSum, signalSum

def corr_lag(x : np.ndarray, y: np.ndarray, fs: float):
    """Calculates correlation with its lags
 
    Args
    ----
    x: np.ndarray
        first singal
    y: np.ndarray
        second signal
    fs: float
        sample rate

    Returns
    -------
    lags and cross-correlation itself
    """

    sigLen = len(x)
    tCC = sig.correlate(x, y, 'same')
    #normDiv = np.sqrt(sig.correlate(x, x, 'same')[int(sigLen/2)] * sig.correlate(y, y, 'same')[int(sigLen/2)])
    tCC /= np.max(np.abs(tCC))

    #tLags = np.linspace(-sigLen/fs, sigLen/fs, sigLen * 2 - 1)
    tLags = np.linspace(-0.5*sigLen/fs, 0.5*sigLen/fs, sigLen)
    tCC = tCC[tLags > 0]
    tLags = tLags[tLags > 0]

    return tLags, tCC

def ca_cfar(x: np.ndarray, trBinSize: int, guBinSize: int, faRate: float, sort: bool = True):
    """ Applies CA-FAR threshold on signal

    Args
    ----
    x: np.ndarray
        signal
    trBinSize: int
        size of the (half) train bin
    guBinSize: int
        size if the (half) guard bin
    faRate: float
        false alarm rate
    sort: bool
        use co-cfar (sorted averaging)

    Returns
    -------
    zero padded threshold
    """

    # ┌-------------┬-----------┬-----------┬-----------┬-------------┐
    # | train bin X | guard bin | candidate | guard bin | train bin X |
    # └-------------┴-----------┴-----------┴-----------┴-------------┘

    x = np.abs(x)
    sigLen = x.size
    binSize = trBinSize + guBinSize

    alpha = trBinSize * 2 * (faRate ** (-1/(trBinSize * 2)) - 1)

    threshList = []

    for i in range(binSize, sigLen - binSize):
        
        leftTrainBin = x[(i-binSize):(i-guBinSize)].copy()
        rightTrainBin = x[(i+guBinSize):(i+binSize)].copy()

        trainBin = leftTrainBin + rightTrainBin

        ### Cell averaging Z = E(X) = µ_x, train bin X
        trBinEst = np.mean(trainBin)
        
        if sort:
            trBinEst = np.median(trainBin)

        estZ = trBinEst * alpha


        threshList.append(estZ)

    return np.pad(threshList, binSize, 'edge')

    #return threshList

def show_butter_bode(saveFig: bool = False):
       
    sys = sig.butter(1, bw / 2, 'lowpass', fs=fs)
    fBode, dBMag, fPha = sig.bode(sys)

    figure = make_subplots(rows=1, cols=2)
    figure.add_trace(go.Scatter(x=fBode, y=dBMag, marker_color='#1F77B4', line=dict(width=3)), row=1, col=1)
    figure.add_trace(go.Scatter(x=fBode, y=fPha, marker_color='#1F77B4', line=dict(width=3)), row=1, col=2)
    figure.update_xaxes(type='log', title_text='Frequency [rad/s]')
    figure.update_yaxes(title_text='Magnitude [dB]', row=1, col=1)
    figure.update_yaxes(title_text='Phase [deg]', row=1, col=2)
    figure.update_layout(title='Butterworth Low-Pass of Order 5 and 20kHz cutoff', showlegend=False)
    figure.show()


    if saveFig:
        figure.write_image("img/bode.pdf", scale=1.5, width=2.5 * DPIEXPORT, height=1 * DPIEXPORT)

def show_raised_cosine(saveFig: bool = False):

    t, filter = flt.rcosfilter(256, rolloff, Tsym, fs)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=t/Tsym, y=filter, marker_color='#000', line=dict(width=3)))
    figure.update_xaxes(title_text='Frequency [Tsym]')
    figure.update_yaxes(title_text='Amplitude')
    figure.show()

    if saveFig:
        figure.write_image("img/cosfir.pdf", scale=1, width=2.5 * DPIEXPORT, height=1 * DPIEXPORT)

def gen_gwn_snr(signal: np.ndarray, stdEst: float, snr: float):

    #stdEst = np.std(signal) / snr
    #stdEst = np.mean(np.std(signals, axis=-1)) / snr

    gwn = np.random.normal(0, stdEst / snr, len(signal))

    return gwn

def simulation(tDelays: list[float], numOfAnchors: int, addGWN = False, startSeed: int = 42, showAll: bool = False, targetSNR: float = 1.0, useSim: bool = True, channel: str = 'CASTLE1', polyDeg: int = 10):

    figure = make_subplots(rows=numOfAnchors, cols=1)
    lstAnchors = []
    lstSignals = []

    global fs, load_index

    load_index = 0
    
    for i in range(startSeed, startSeed + numOfAnchors):

        ### generates gold code
        bCode = gold_seq(polyDeg, i, 2)[0]
        #bCode = np.random.uniform(-1, 1, codeLen)

        ### upsample and filter via FIR cosine
        filter = flt.rcosfilter(1024, rolloff, Tsym, fs)[1]
        tSigBB = sig.resample_poly(bCode, SpS, 1, window=filter)

        ### time axis of upsampled signal
        codeLen = 2 ** polyDeg + 1
        time, tstep = np.linspace(0, Tsym * codeLen, len(tSigBB), retstep=True)

        ### use baseband signal for correlation later as anchor
        lstAnchors.append(tSigBB)

        ### shift spectrum to transmission band and save it for simulation
        time, tSigTB = gen_tb_signal(time, tSigBB, False, polyDeg)
        
        ### use simulation signal or just passthrough the generated transfer-band signal
        if useSim:
            tSigTBr = get_tb_signal(len(tSigTB), channel, polyDeg)
            time = genAxis(len(tSigTBr), tstep)
        else:
            tSigTBr = np.real(tSigTB)

        ### use recv transfer-band singal for sum
        lstSignals.append(tSigTBr)

    ### sum up all signals with added delay zero padding and extenden length
    timeSum, tSigTBrSum = delay_sum(lstSignals, tDelays, fs, tstep)

    ### setting win length resulting in a guard bin size of peak width 0.00014
    guLen = int(fs * 0.00014)

    ### add white noise
    if addGWN:
        stdEst = np.mean(np.std(lstSignals, axis=-1))
        gwn = gen_gwn_snr(tSigTBrSum, stdEst, targetSNR)

        tSigTBrSum += gwn

        print("SNR_dB:", 10 * np.log10(stdEst/np.std(gwn)), "dB")
        print("SNR:", stdEst/np.std(gwn))

    ### applying SysOrd order butterworth bandpass forwards and backwards
    b, a = sig.butter(sysOrd, [fc - bw/2, fc + bw/2], 'bandpass', fs=fs, output='ba')

    #figure = go.Figure()
    #figure.add_trace(go.Scatter(y=np.real(tSigTBrSum)))
    #figure.show()

    tSigTBrSum = sig.filtfilt(b, a, tSigTBrSum)

    #figure = go.Figure()
    #figure.add_trace(go.Scatter(y=np.real(tSigTBrSum)))
    #figure.show()

    ### shift sum of signals to baseband
    tSigBBrSum = tSigTBrSum * np.exp(-2.j*np.pi*fc*timeSum)

    ### applying SysOrd order butterworth low pass forwards and backwards
    b, a = sig.butter(sysOrd, bw/2, 'lowpass', fs=fs, output='ba')
    tSigBBrSum = sig.filtfilt(b, a, tSigBBrSum)

    figure = make_subplots(rows=numOfAnchors, cols=1)
    index = 1

    estDelays = []
    lstSigCCpks = []

    for si in lstAnchors:

        si = np.append(si, [0] * (len(tSigBBrSum) - len(si)))
        tauCC, tauSigCC = corr_lag(tSigBBrSum, si, fs)
        varSigSum = ca_cfar(tauSigCC, guLen * 6, guLen, 1.2e-1, sort=False)

        SigCCpks = np.abs(tauSigCC.copy())
        #SigCCpks[SigCCpks < varSigSum] = 0
        #SigCCpks[SigCCpks > varSigSum] = 1
        #SigCCpks = SigCCpks.astype(dtype=bool)
        sigCCind= np.where(SigCCpks > varSigSum)
        #lstSigCCpks.append(SigCCpks)

        lagInd = np.argmax(abs(tauSigCC))

        print("indexes: ", lagInd)
        estDelays.append(tauCC[lagInd])
        figure.add_trace(go.Scatter(x=tauCC, y=np.abs(tauSigCC), mode='lines', marker_color='#000'), row=index, col=1)
        figure.add_trace(go.Scatter(x=tauCC, y=varSigSum, mode='lines', marker_color='#636EFA'), row=index, col=1)

        lastVal = 0
        for i in sigCCind[0]:
            if i != lastVal + 1:
                figure.add_vline(tauCC[i], line_color='#00CC96', line_width=3, line_dash='dash', row=index, col=1)
                lstSigCCpks.append(tauCC[i])
            lastVal = i

            #figure.add_vrect()

        figure.add_vline(tauCC[lagInd], line_color='#EF553B', line_width=3, row=index, col=1)

        
        
        index += 1

    fig_title = "code degree: " + str(polyDeg) + ", watermark channel: " + channel + ", target SNR: " + str(10*np.log10(abs(targetSNR))) + "dB"
    figure.update_layout(showlegend=False, title=fig_title)
    if showAll:
        figure.show() 

    """figure = make_subplots(rows=numOfAnchors, cols=1)
    index = 1
    for si in lstSigCCpks:
        figure.add_trace(go.Scatter(x=tauCC, y=si, mode='lines', marker_color='#000'), row=index, col=1)
        index += 1

    fig_title = "code degree: " + str(deg) + ", watermark channel: " + channel + ", target SNR: " + str(10*np.log10(abs(targetSNR))) + "dB"
    figure.update_layout(showlegend=False, title=fig_title)
    figure.show() """
    

    if showAll:

        fBB, fSigBB = _get_fftfunc(tSigBB, fs)
        fTB, fSigTB = _get_fftfunc(tSigTB, fs)
        fBBr, fSigBBrSum = _get_fftfunc(tSigBBrSum, fs)

        print("total signal length is", round(timeSum[-1], 5), "s")

        ### code to transfer band process plots and backwards process plots
        figure = make_subplots(rows=3, cols=1)
        figure.add_trace(go.Bar(y=bCode, name="bit domain", marker_color='#19D3F3'), row=1, col=1)
        figure.add_trace(go.Scatter(x=time, y=np.real(tSigBB), mode='lines', name="baseband signal", marker_color='#EF553B'), row=2, col=1)
        figure.add_trace(go.Scatter(x=time, y=np.real(tSigTB), mode='lines', name="carrier signal", marker_color='#636EFA'), row=2, col=1)
        figure.add_trace(go.Scatter(x=fBB, y=np.abs(fSigBB), fill='tozeroy', mode='lines', name="baseband", marker_color='#EF553B'), row=3, col=1)
        figure.add_trace(go.Scatter(x=fTB, y=np.abs(fSigTB), fill='tozeroy', mode='lines', name="carrier", marker_color='#636EFA'), row=3, col=1)
        figure.update_layout(title='Baseband & Transmission-Band of Send Signal')
        figure.show()
 
        figure = make_subplots(rows=2, cols=1)
        figure.add_trace(go.Scatter(x=timeSum, y=np.real(tSigBBrSum), mode='lines', name="recv baseband signal", marker_color='#FFA51A'), row=1, col=1)
        figure.add_trace(go.Scatter(x=fBBr, y=np.abs(fSigBBrSum), fill='tozeroy', mode='lines', name="recv baseband", marker_color='#FFA51A'), row=2, col=1)
        figure.update_layout(title='Recieved Sum of Signals in Baseband')
        figure.show()

    relDelays = (estDelays[0] - estDelays[1], estDelays[0] - estDelays[2], estDelays[0] - estDelays[3])

    return relDelays, lstSigCCpks

def main():
    #for snr in [-20, -15, -10, -5, 0, 5, 10, 15, 20]:
    snr = 0
    snr = 10 ** (snr / 10)
    simulation([10e-3, 20e-3, 30e-3, 40e-3], 4, showAll=False, addGWN=True, targetSNR=snr, useSim=False, polyDeg=8)
    #show_butter_bode(saveFig=True)
    #show_raised_cosine(saveFig=True)

if __name__ == "__main__":
    main()
    
# SNR tp error(localtion measured - real location) and different degs/CHANNELS
# use SNR for unsummed signals
# generate delay list for circular path of 3 anchors (time difference of arrival)
# pdf Fr 12:00


"""def main():

    startSeed = 32

    numOfAnchors = 5

    figure = make_subplots(rows=numOfAnchors, cols=1)
    tSendSig = []

    tDelays = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3]

    for i in range(startSeed, startSeed + numOfAnchors):
        rawCode = gold_seq(deg, i, 2)[0]
        signal = gen_TB_signal(rawCode)
        signal = get_BB_signal(signal[0], signal[1])
        tSendSig.append(signal[1])
        figure.add_trace(go.Scatter(x=signal[0], y=signal[1], mode='lines', marker_color='#000'), row=i-(startSeed-1), col=1)

    figure.show() 

    numOfCorr = comb(numOfAnchors, 2)

    figure = make_subplots(rows=numOfCorr, cols=2)

    index = 1
    for pair in combinations(tSendSig, 2):
        SigCC = sig.correlate(pair[0], pair[1], 'same')
        tauCC = sig.correlation_lags(len(pair[0]), len(pair[1]), 'same')
        figure.add_trace(go.Scatter(x=tauCC, y=SigCC, mode='lines', marker_color='#000'), row=index, col=1)
        index += 1

    index = 1
    for si in tSendSig:
        SigAC = sig.correlate(si, si, 'same')
        tauAC = sig.correlation_lags(len(si), len(si), 'same')
        figure.add_trace(go.Scatter(x=tauAC, y=SigAC, mode='lines', marker_color='#000'), row=index, col=2)
        index += 1
    figure.update_layout(showlegend=False)
    figure.show() 

    figure = make_subplots(rows=1, cols=1)
    tSigSum = delay_sum(tSendSig, tDelays, fs)
    figure.add_trace(go.Scatter(x=tSigSum[0], y=tSigSum[1], mode='lines', marker_color='#000'), row=1, col=1)
    figure.update_layout(showlegend=False)
    figure.show() 

    figure = make_subplots(rows=numOfAnchors, cols=1)
    index = 1
    for si in tSendSig:
        si = np.append(si, [0] * (len(tSigSum[1]) - len(si)))
        SigCC = corr_lag(tSigSum[1], si, fs)
        figure.add_trace(go.Scatter(x=SigCC[0], y=SigCC[1], mode='lines', marker_color='#000'), row=index, col=1)
        lagInd = np.argmax(SigCC[1])
        figure.add_vline(SigCC[0][lagInd], line_color='#EF553B', line_width=3, line_dash='dash', row=index, col=1)
        index += 1
    figure.update_layout(showlegend=False)
    figure.show() """