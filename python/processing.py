import numpy as np
import plotly.graph_objects as go
from commpy import filters as flt
from plotly.subplots import make_subplots
from scipy import signal as sig
from scipy.io import savemat, loadmat, wavfile
from sequences import gold_seq

DPIEXPORT = 400
MATSAVE = '/Users/sk/Library/CloudStorage/OneDrive-Persönlich/Studium/TUHH/3. Semester Master/Forschungsprojekt/uw-watermark-main/Watermark/input/signals/tSigTB_'
WAVLOAD = '/Users/sk/Library/CloudStorage/OneDrive-Persönlich/Studium/TUHH/3. Semester Master/Forschungsprojekt/uw-watermark-main/Watermark/output/'
CHANNEL = 'CASTLE3'

file_index = 0
load_index = 0

### signal parameters
fs = 200e3          # sampling rate
bw = 20e3           # bandwith
Tsym = 1 / bw       # symbol length
SpS = fs * Tsym     # upsampling factor
rolloff = 1/8       # FIR cosine filter coefficent
fc = 62.5e3         # carrier freqency
sysOrd = 5          # order of butterworth filter

targetSNR = 0.1        # targeted Signal Noise Ratio for addtive GWN generator
watermarkDelay = 12e-3  # delay of watermark simulation time of flight

### gold sequence generation
deg = 10
codeLen = 2 ** deg + 1

""" def testPlotting(sig, time, num):
    fig = make_subplots(rows=2, cols=1)
    print(len(sig))
    fig.add_trace(go.Scatter(y=sig, x=time), row=1, col=1)
    fSigT, freqT = _get_fftfunc(sig, fs)
    fig.add_trace(go.Scatter(y=freqT, x=np.abs(fSigT)), row=2, col=1)
    mytitle = "Test " + str(num)
    fig.update_layout(title=mytitle)
    fig.show() """

def sig_noise_ratio(sig: np.ndarray, noise: np.ndarray):
    #https://de.mathworks.com/help/signal/ref/mag2db.html
    #https://de.mathworks.com/help/signal/ref/rssq.html
    snr = np.std(sig) / np.std(noise)
    return snr

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

def gen_tb_signal(time: np.ndarray, tSigBB: np.ndarray, saveMat: bool = True):
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

    global file_index

    ### shift spectrum to transmission band
    tSigTB = np.real(tSigBB * np.exp(2.j*np.pi*fc*time))

    ### save signal as matlab object for watermark
    matobj = {'fs_x': fs, 'nBits': 0, 'x': tSigTB}
    if saveMat:
        savemat(MATSAVE + "d" + str(deg) + "_" + str(file_index) + '.mat', matobj)
    file_index += 1

    return (time, tSigTB)

def gen_bb_signal(time: np.ndarray, tSigTB: np.ndarray):
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

    filepath = CHANNEL + "/sigTB_" + "d" + str(deg) + "_" + str(load_index) + "/" + CHANNEL + "_001.wav"

    sigLen = len(tSigTB)
    _, tSigTBr = wavfile.read(WAVLOAD + filepath)
    
    waveLen = loadmat(WAVLOAD + CHANNEL + "/sigTB_" + "d" + str(deg) + "_" + str(load_index) + "/bookkeeping.mat")
    #waveLen = waveLen['bk'][0][0][2][0][1]
    tSigTBr = tSigTBr[(watermarkDelay * fs):(sigLen + watermarkDelay * fs)]
    load_index += 1

    ### add white noise to not delayed part
    gwn = gen_gwn_snr(tSigTBr, targetSNR)
    tSigTBr += gwn

    #testPlotting(np.real(tSigTB), time, 2)
    print("SNR_dB: ", 10 * np.log10(sig_noise_ratio(tSigTB - gwn, gwn)), "dB")
    print("SNR", sig_noise_ratio(tSigTB - gwn, gwn))

    ### reshift spectrum to baseband
    tSigBBr = tSigTBr * np.exp(2.j*np.pi*-fc*time)

    return (time, tSigBBr)

def delay_sum(signals: list[np.ndarray], delays: list[float], fs: float):
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

    signalSum = np.zeros(singalSumLen)
    timeSum = np.linspace(0, Tsym * codeLen, singalSumLen)
    
    for s, d in zip(signals, delays):

        index = range(int(np.floor(d * fs) + 1), int((np.floor(d * fs) + len(s) + 1)))
        signalSum[index] = signalSum[index] + s

    return (timeSum, signalSum)

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
    tCC /= np.max(tCC)

    tLags = np.linspace(-0.5 * sigLen/fs, 0.5 * sigLen/fs, sigLen)
    tCC = tCC[tLags > 0]
    tLags = tLags[tLags > 0]

    return (tLags, tCC)

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

    # ┌-----------┬-----------┬-----------┬-----------┬-----------┐
    # | train bin | guard bin | candidate | guard bin | train bin |
    # └-----------┴-----------┴-----------┴-----------┴-----------┘

    x = np.abs(x)
    sigLen = x.size
    binSize = trBinSize + guBinSize

    alpha = trBinSize * 2 * (faRate ** (-1/(trBinSize * 2)) - 1)

    threshList = []

    for i in range(binSize, sigLen - binSize):

        bin = x[i-binSize:i+binSize+1]
        guard = x[i-guBinSize:i+guBinSize+1]

        #if sort:
        #    bin = np.sort(bin)
        #    guard = np.sort(bin)[(len(bin) - guBinSize):(len(bin) + guBinSize)]
        # ignore guard cells for sorte averaging

        binSum = np.sum(bin)
        guBinSum = np.sum(guard)

        estThresh = (binSum - guBinSum) / (trBinSize * 2)
        estThresh *= alpha
        threshList.append(estThresh)

    return np.pad(threshList, (binSize, sigLen - binSize), 'edge')

def show_butter_bode(saveFig: bool = False):
       
    sys = sig.butter(sysOrd, bw, 'lowpass', fs=fs)
    fBode, dBMag, fPha = sig.bode(sys)

    figure = make_subplots(rows=1, cols=2, x_title='Frequency [deg]', subplot_titles=('Magnitude [dB]', 'Phase [deg]'))
    figure.add_trace(go.Scatter(x=fBode, y=dBMag, fill='tozerox', marker_color='#EF553B'), row=1, col=1)
    figure.add_trace(go.Scatter(x=fBode, y=fPha, fill='tozerox'), row=1, col=2)
    figure.update_xaxes(type='log')
    figure.update_layout(title='Butterworth Low-Pass of Order 5 and 20kHz cutoff', showlegend=False)
    figure.show()

    if saveFig:
        figure.write_image("img/bode.pdf", scale=1, width=2.5 * DPIEXPORT, height=1 * DPIEXPORT)

def show_raised_cosine(saveFig: bool = False):

    filter = flt.rcosfilter(1024, rolloff, Tsym, fs)[1]
    figure = go.Figure()
    figure.add_trace(go.Scatter(y=filter[388:650], marker_color='#000'))
    figure.show()

    if saveFig:
        figure.write_image("img/cosfir.pdf", scale=1, width=2.5 * DPIEXPORT, height=1 * DPIEXPORT)

def gen_gwn_snr(signal: np.ndarray, snr: float):

    stdEst = np.std(signal) / snr
    gwn = np.random.normal(0, stdEst, len(signal))

    return gwn

def simulation(tDelays: list[float], numOfAnchors: int, addGWN = False, startSeed: int = 42, showAll: bool = False):

    figure = make_subplots(rows=numOfAnchors, cols=1)
    tSendSig = []
    
    for i in range(startSeed, startSeed + numOfAnchors):

        ### generates gold code
        bCode = gold_seq(deg, i, 2)[0]
        #rawCode = np.random.uniform(-1, 1, codeLen)

        ### upsample and filter via FIR cosine
        filter = flt.rcosfilter(1024, rolloff, Tsym, fs)[1]
        tSigBB = sig.resample_poly(bCode, SpS, 1, window=filter)

        ### create fitting time axis
        time = np.linspace(0, Tsym * codeLen, len(tSigBB))

        ### shift spectrum to transmission band and save it for simulation
        time, tSigTB = gen_tb_signal(time, tSigBB)

        ### shift back to baseband loaded matlab wavefile from simulation
        time, tSigBBr = gen_bb_signal(time, tSigTB)

        ### applying SysOrd order butterworth low pass forwards and backwards
        b, a = sig.butter(sysOrd, bw, 'lowpass', fs=fs, output='ba')
        tSigBBr = sig.filtfilt(b, a, tSigBBr)

        tSendSig.append(tSigBBr)

    tSigSum = delay_sum(np.real(tSendSig), tDelays, fs)
    winLen = fs * 5e-4

    """figure = make_subplots(rows=numOfAnchors, cols=1)
    index = 1
    for si in tSendSig:

        si = np.append(si, [0] * (len(tSigSum[1]) - len(si)))
        SigCC = corr_lag(np.real(tSigSum[1]), np.real(si), fs)
        varSigSum = ca_cfar(SigCC[1], int(winLen), int(0.1 * winLen), 1e-3, sort=False)
        lagInd = np.argmax(SigCC[1])

        figure.add_trace(go.Scatter(x=SigCC[0], y=SigCC[1], mode='lines', marker_color='#000'), row=index, col=1)
        figure.add_trace(go.Scatter(x=SigCC[0], y=varSigSum, mode='lines', marker_color='#636EFA'), row=index, col=1)
        figure.add_vline(SigCC[0][lagInd], line_color='#EF553B', line_width=3, line_dash='dash', row=index, col=1)
        

        index += 1

    fig_title = "code degree: " + str(deg) + ", watermark channel: " + CHANNEL + ", target SNR: " + str(10*np.log10(abs(targetSNR))) + "dB"
    figure.update_layout(showlegend=False, title=fig_title)
    
    figure.show() 

    if showAll:

        fBB, fSigBB = _get_fftfunc(tSigBB, fs)
        fTB, fSigTB = _get_fftfunc(tSigTB, fs)
        fBBr, fSigBBr = _get_fftfunc(tSigBBr, fs)

        print("signal length in s", len(tSigTB) / fs)


        # remove delay and extended time if recieved baseband imported from watermark
        if usesim:
            time = time[len(time)//2:]
            tSigBBr = tSigBBr[int(tDelays[-1] * fs):]

        ### code to transfer band process plots and backwards process plots
        figure = make_subplots(rows=3, cols=1)
        figure.add_trace(go.Bar(y=rawCode, name="bit domain", marker_color='#19D3F3'), row=1, col=1)
        figure.add_trace(go.Scatter(x=time, y=tSigBB, mode='lines', name="baseband signal", marker_color='#EF553B'), row=2, col=1)
        figure.add_trace(go.Scatter(x=time, y=tSigTB, mode='lines', name="carrier signal", marker_color='#636EFA'), row=2, col=1)
        figure.add_trace(go.Scatter(x=time, y=np.real(tSigBBr), mode='lines', name="recv baseband signal", marker_color='#FFA51A'), row=2, col=1)
        figure.add_trace(go.Scatter(x=fBB, y=np.abs(fSigBB), fill='tozeroy', mode='lines', name="baseband", marker_color='#EF553B'), row=3, col=1)
        figure.add_trace(go.Scatter(x=fTB, y=np.abs(fSigTB), fill='tozeroy', mode='lines', name="carrier", marker_color='#636EFA'), row=3, col=1)
        figure.add_trace(go.Scatter(x=fBBr, y=np.abs(fSigBBr), fill='tozeroy', mode='lines', name="recv baseband", marker_color='#FFA51A'), row=3, col=1)
        figure.update_layout(title='Baseband & Transmission-Band Processing')
        figure.show()

        # TODOS: 
        # SNR verdammt nochmal hinbekommen
        # simulation so feststellen, dass singal ohne delay genommen wird
        # kombinationen mit SNR festmachen"""
        


def main():
    simulation([20e-3, 30e-3, 40e-3], 3, showAll=False, addGWN=True)
    #showButterBode()
    #showRaisedCosine()

if __name__ == "__main__":
    main()
    



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