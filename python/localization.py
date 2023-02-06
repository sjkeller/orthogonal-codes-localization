from processing import simulation, corr_lag, ca_cfar
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

SOFSOUND_WATER = 1500
POSITIONS = 15

def remove_under_threshold(a: list[int], threshold: int):    
    """
    Given a list of values 'a' and a threshold value, the function returns a new list 'b' that only contains elements
    from 'a' where the difference between the current and previous elements is greater than the threshold.
    The first element of 'a' is always included in 'b'.
    
    Args
    ----
    a: list[int]
        A list of numerical values
    threshold: int
        threshold value
    
    Returns
    -------
    A new list of values from 'a' where the difference between the current and previous elements is greater than 'threshold'
    
    """
    a = np.array(a)
    b = [a[0]]
    diff = np.diff(a)
    for i in range(len(a) - 1):
        if diff[i] > threshold:
            b.append(a[i + 1])
    return b

def remove_sublists(lists: list[list], distance: int):
    """ Removes sublists with smaller difference in start and endvalue

    example: x
        remove_sublists([[1,2,3],[7,8,9],[10,11,12,13]], 3) => [[1,2,3],[7,8,9]]
        last sublist got removed because 10 - 9 < distance

    Args
    ----
    lists: list[list]
        list of sublists containing consecutive values
    distance: int
        threshold for removing sublists

    Returns
    -------
    list containing of wanted sublists
    """
    new_lists = []
    for i in range(len(lists)-1):
        first_sublist = lists[i]
        second_sublist = lists[i+1]
        if abs(first_sublist[-1] - second_sublist[0]) > distance:
            new_lists.append(first_sublist)
    new_lists.append(lists[-1])
    return new_lists

def group_consecutive_values(input_list: list[int], threshold: int = 1):
    """ Groups consecutive values and put them as sublists into a list

    Args
    ----
    input_list: list[int]
        list of integers in order 0 ... len(input_list) - 1
    threshold:
        used for removing certain consecutive groups, set to 1 by default

    Returns
    -------
    list of consecutive sublists
    """
    if len(input_list) == 0:
        return []
    result = []
    current_group = [input_list[0]]
    for i in range(1, len(input_list)):
        if input_list[i] == input_list[i-1] + 1:
            current_group.append(input_list[i])
        else:
            result.append(current_group)
            current_group = [input_list[i]]
    result.append(current_group)
    result = remove_sublists(result, threshold)
    return result

def totalError(listRealPos: list[tuple], listEstPos: list[tuple]):
    errors = []
    for i in range(len(listRealPos)):
        realPos = np.array(listRealPos[i])
        estPos = np.array(listEstPos[i])
        errors.append(np.linalg.norm(estPos - realPos))

    return errors

def quadraticApprox(anchorPos: list[tuple], toas: list[float], lastZPos: float = 0.0):
    """ TDOA Alogrithm for localaization by four anchors 

    Args
    ----
    anchorPos: list[tuple]
        anchors positions in 3d space
    toas: list[float]
        time of arrivals in seconds
    zBound: float
        lower threshold for z candidate

    Returns
    -------
    position of target in 3d space
    """

    d01 = SOFSOUND_WATER * (toas[0] - toas[1])
    d02 = SOFSOUND_WATER * (toas[0] - toas[2])
    d21 = SOFSOUND_WATER * (toas[2] - toas[1])
    d23 = SOFSOUND_WATER * (toas[2] - toas[3])

    print("distances:", d01, d02, d21, d23)

    x0, y0, z0 = anchorPos[0]
    x1, y1, z1 = anchorPos[1]
    x2, y2, z2 = anchorPos[2]
    x3, y3, z3 = anchorPos[3]

    x10 = x1 - x0
    x20 = x2 - x0
    x12 = x1 - x2
    x32 = x3 - x2

    y10 = y1 - y0
    y20 = y2 - y0
    y12 = y1 - y2
    y32 = y3 - y2

    z10 = z1 - z0
    z20 = z2 - z0
    z12 = z1 - z2
    z32 = z3 - z2

    varA = (d02 * x10 - d01 * x20) / (d01 * y20 - d02 * y10)
    varB = (d02 * z10 - d01 * z20) / (d01 * y20 - d02 * y10)
    varC = (d02 * (d01 ** 2 + x0 ** 2 - x1 ** 2 + y0 ** 2 - y1 ** 2 + z0 ** 2 - z1 ** 2) - d01 * (d02 ** 2 + x0 ** 2 - x2 ** 2 + y0 ** 2 - y2 ** 2 + z0 ** 2 - z2 ** 2)) / (2 * (d01 * y20 - d02 * y10))
    
    varD = (d23 * x12 - d21 * x32) / (d21 * y32 - d23 * y12)
    varE = (d23 * z12 - d21 * z32) / (d21 * y32 - d23 * y12)
    varF = (d23 * (d21 ** 2 + x2 ** 2 - x1 ** 2 + y2 ** 2 - y1 ** 2 + z2 ** 2 - z1 ** 2) - d21 * (d23 ** 2 + x2 ** 2 - x3 ** 2 + y2 ** 2 - y3 ** 2 + z2 ** 2 - z3 ** 2)) / (2 * (d21 * y32 - d23 * y12))

    varG = (varE - varB) / (varA - varD)
    varH = (varF - varC) / (varA - varD)
    varI = varA * varG + varB
    varJ = varA * varH + varC

    varK = d02 ** 2 + x0 ** 2 - x2 ** 2 + y0 ** 2 - y2 ** 2 + z0 ** 2 - z2 ** 2 + 2 * x20 * varH + 2 * y20 * varJ
    varL = 2 * (x20 * varG + y20 * varI + z20) # was written wrong in paper 
    varM = 4 * d02 ** 2 * (varG ** 2 + varI ** 2 + 1) - varL ** 2
    varN = 8 * d02 ** 2 * (varG * (x0 - varH) + varI * (y0 - varJ) + z0) + 2 * varL * varK
    varO = 4 * d02 ** 2 * ((x0 - varH) ** 2 + (y0 - varJ) ** 2 + z0 ** 2) - varK ** 2

    pqFt = varN / (2 * varM)
    pqSd = (varN /(2 * varM)) ** 2 - (varO / varM)

    #print("pq-sel", pqFt, pqSd)
    zposCand1 = pqFt + np.sqrt(np.abs(pqSd))
    zposCand2 = pqFt - np.sqrt(np.abs(pqSd))

    #print("z candidates", zposCand1, zposCand2)

    #zpos = zposCand1 if zposCand1 < 0 else zposCand2
    #zpos = np.min([np.max([zposCand1, zposCand2, zBound]), 0])

    ### check which z position might be more accurate depening on last one
    if abs(zposCand1 - lastZPos) < abs(zposCand2 - lastZPos):
        zpos = zposCand1
    else:
        zpos = zposCand2

    ypos = varI * zpos + varJ
    xpos = varG * zpos + varH

    return xpos, ypos, zpos
    #return xpos, ypos, zposCand1, zposCand2

def dist_to_delay(achorPos: list[tuple], vehPos: tuple[float], absolute: bool = False):
    """ Converts 3d distances to TDOAs ot TOAs

    Args
    ----
    anchorsPos: list[tuple]
        absolute 3d Positions of four anchors as (x,y,z)
    vehPos: tuple[float]
        absolute 3d positions of undervater vehicle/target
    absolute: bool
        returns TOAs instead of TDOAs if True, False by default

    Returns
    -------
    all four TOAs or three TDOAs
    """

    # tau_i = 1/c * (|S - S_0| - |S - S_i|)
    s = np.array(vehPos)
    s0 = np.array(achorPos[0])
    s1 = np.array(achorPos[1])
    s2 = np.array(achorPos[2])
    s3 = np.array(achorPos[3])
    #s4 = np.array(achorPos[4])

    dist0 = np.linalg.norm(s - s0)
    dist1 = np.linalg.norm(s - s1)
    dist2 = np.linalg.norm(s - s2)
    dist3 = np.linalg.norm(s - s3)
    #dist4 = np.linalg.norm(s - s4)

    # S_0 |-----^------------------------|
    # S_1 |-------------^----------------|
    # S_2 |-----------------------^------|
    # tau       0       1         2
    # tau_0i = tau_i - tau_0  

    if absolute:
        t0 = dist0 / SOFSOUND_WATER
        t1 = dist1 / SOFSOUND_WATER
        t2 = dist2 / SOFSOUND_WATER
        t3 = dist3 / SOFSOUND_WATER
        #t4 = dist4 / SOFSOUND_WATER

        return t0, t1, t2, t3

    else:
        tau01 = (dist0 - dist1) / SOFSOUND_WATER
        tau02 = (dist0 - dist2) / SOFSOUND_WATER
        tau03 = (dist0 - dist3) / SOFSOUND_WATER
        #tau04 = (dist0 - dist4) / SOFSOUND_WATER

        return tau01, tau02, tau03

def circlePath(scale: float, offset: float, res: int = 10):
    """
    This function calculates the path of a circle in 3D space.

    Args
    ----
    scale: float
        scaling factor for the circle.
    offset: float
        offset for the x and y values.
    res: int
        number of resolution points along the path, default is 10.

    Returns
    -------
    x, y, z: 1D arrays of floats, the x, y, and z coordinates of the circle path.

    """
    phi = np.linspace(0, 3*np.pi, res, endpoint=False)
    x = scale * np.sin(phi) + offset
    y = scale * np.cos(phi) + offset
    z = np.array(range(-res, -1))
    return x,y,z

def underwater_localization(anchorsPos: list[tuple], path: tuple[np.ndarray], deg: int = 10, snr: (float | None) = None, simChannel: (str | None) = None):
    """Localization of underwater target via tdoa

    Args
    ----
    anchorsPos: list[tuple]
        absolute 3d Positions of four anchors as (x,y,z)
    path: tuple[np.ndarray]
        list of path coorindates as (x_i ... x_n, y_i ... y_n, z_0 ... z_n)
    deg: int
        degree of pseudorandom code, which influences its total length
    snr: float
        target snr in dB for additive white gaussian noise
    simChannel: str
        channel name which should be used for simulation

    Returns
    -------
    - list of estimated positions
    - list of cfar localization data as (time axis, correlation sum, cfar threshold)
    - tuple of raw recieved signal seperated in its time axis and itself
    - error between real and eastimatet positions
    """

    lstRealPos = []
    lstEstPos = []

    xPos, yPos, zPos = path
    POSITIONS = len(xPos)

    xAnchors = list(zip(*anchorsPos))[0]
    yAnchors = list(zip(*anchorsPos))[1]
    zAnchors = list(zip(*anchorsPos))[2]
    if snr is not None:
        snr = 10 ** (snr / 10)
    fs = 200e3

    totalSumTime = []
    totalSumSignals = []
    lstAnchors = []

    for cx, cy, cz in zip(xPos, yPos, zPos):
        
        absDelays = dist_to_delay(anchorsPos, (cx, cy, cz), absolute=True)
        lstRealPos.append((cx, cy, cz))

        if snr is not None:
            if simChannel:
                _, tSigSum, lstAnchors = simulation(absDelays, 4, showAll=False, addGWN=True, useSim=True, channel=simChannel, polyDeg=deg, targetSNR=snr)
            else:
                _, tSigSum, lstAnchors = simulation(absDelays, 4, showAll=False, addGWN=True, useSim=False, polyDeg=deg, targetSNR=snr)
        else:
            _, tSigSum, lstAnchors = simulation(absDelays, 4, showAll=False, addGWN=False, useSim=False, polyDeg=deg)


        #else:
        #    delays, tSigSum, lstAnchors = simulation(absDelays, 4, showAll=False, addGWN=False, useSim=False, polyDeg=deg)

        totalSumSignals = np.append(totalSumSignals, tSigSum[1])

    totalSumTime = np.linspace(0, len(totalSumSignals) / fs, len(totalSumSignals))

    guLen = int(200e3 * 0.00014)
    anchorPeaks = []
    lstSumDelays = []
    lstCFARSig = []

    for i in range(len(lstAnchors)):
        peaks = []

        anchor = np.append(lstAnchors[i], [0] * (len(totalSumSignals) - len(lstAnchors[i])))
        tauCC, tauSigCC = corr_lag(totalSumSignals, anchor, fs) 
        #cfarSigSum = ca_cfar(tauSigCC, guLen * 3, guLen, 1.0e-1, sort=False, threshold=0.2)
        #sigCCpks = np.abs(tauSigCC.copy())
        
        groups = []
        
        if deg == 8:
            cfarStartThresh = 0.172
        if deg == 9:
            cfarStartThresh = 0.16
        if deg == 10:
            cfarStartThresh = 0.086
            #cfarStartThresh = 0.07

        
        
        cfarSigSum = ca_cfar(tauSigCC, guLen * 6, guLen, cfarStartThresh, sort=False, threshold=0.2)
        sigCCpks = np.abs(tauSigCC.copy())
        sigCCInd = np.where(sigCCpks > cfarSigSum)
        groups = group_consecutive_values(sigCCInd[0])
        lstCFARSig.append((tauCC, tauSigCC, cfarSigSum))
        
        for gr in groups:
            # get maximum of every index group
            cfarPeakInd = np.where(np.abs(tauSigCC) == np.max(np.abs(tauSigCC[gr])))[0][0]
            #peaks.append(tauCC[cfarPeakInd])
            peaks.append(cfarPeakInd)

        print("all peaks", peaks)
        peaks = remove_under_threshold(peaks, int(fs * 0.01))
        print("filt peaks", peaks)
        
        for i in range(len(peaks)):
            peaks[i] = tauCC[peaks[i]]
        
        """if deg == 8:
            cfarStartThresh = 0.172
        if deg == 9:
            cfarStartThresh = 0.118
        if deg == 10:
            cfarStartThresh = 0.086



        groups = [0] * (POSITIONS + 1)
        print("selecting peaks for anchor", i)
        lastlen = 0
        while len(groups) > POSITIONS - 1:
            lastlen = len(groups)
            print(len(groups), end=",")
            
            
            cfarStartThresh -= 0.02e-1

            cfarSigSum = ca_cfar(tauSigCC, guLen * 6, guLen, cfarStartThresh, sort=False, threshold=np.mean(np.abs(tauSigCC)))
            sigCCInd = np.where(sigCCpks > cfarSigSum)
            groups = group_consecutive_values(sigCCInd[0])


        
        if len(groups) != POSITIONS - 1:
            
            cfarSigSum = ca_cfar(tauSigCC, guLen * 6, guLen, cfarStartThresh + 0.02e-1, sort=False, threshold=np.mean(np.abs(tauSigCC)))
            print("winning thresh", cfarStartThresh + 0.02e-1)
            sigCCInd = np.where(sigCCpks > cfarSigSum)
            groups = group_consecutive_values(sigCCInd[0])
        else:
            print("winning thresh", cfarStartThresh)


        print("selected", len(groups), "peak groups")

        lstCFARSig.append((tauCC, tauSigCC, cfarSigSum))
            
        cfarSigSum = ca_cfar(tauSigCC, guLen * 4, guLen, 2.1e-1, sort=False, threshold=0.55)
        
        lstCFARSig.append((tauCC, tauSigCC, cfarSigSum))

        sigCCpks = np.abs(tauSigCC.copy())
        sigCCInd = np.where(sigCCpks > cfarSigSum)

        # group consecutive values for further processing
        groups = group_consecutive_values(sigCCInd[0])

        if len(groups) != POSITIONS - 1:
            print("len groups", len(groups), POSITIONS - 1)
            print("Peaks not succefully detected!")
        lastgr = [0]
        for gr in groups:
            # get maximum of every index group
            #if gr[0] > (lastgr[-1] + int(fs * 0.01)):
            cfarPeakInd = np.where(np.abs(tauSigCC) == np.max(np.abs(tauSigCC[gr])))[0][0]
            lastgr = gr
            peaks.append(tauCC[cfarPeakInd])"""
        
        anchorPeaks.append(peaks)
        
    lastZPos = -1.0
    print("pks", anchorPeaks)
    for pos in range(POSITIONS - 1):
        tdoaSelection = [item[pos] for item in anchorPeaks]
        #lstSumDelays.append(tdoaSelection)
        estPos = quadraticApprox(anchorsPos, tdoaSelection, lastZPos)
        lastZPos = estPos[2]
        lstEstPos.append((estPos[0], estPos[1], estPos[2]))

    posError = totalError(lstRealPos, lstEstPos)

    #for pos in range(POSITIONS - 1):
    #    estPos = quadraticApprox(anchorsPos, lstSumDelays[pos])
    #    lstEstPos.append((estPos[0], estPos[1], estPos[2]))

    #xEstPos = list(zip(*lstEstPos))[0]
    #yEstPos = list(zip(*lstEstPos))[1]
    #zEstPos = list(zip(*lstEstPos))[2]

    return lstEstPos, anchorPeaks, lstCFARSig, (totalSumTime, totalSumSignals), posError

def main():
    POS = 20

    anc = [(15,1,7),(200,10,5),(195,210,6),(16,190,3)]
    testPath = circlePath(30, 100, POS + 1)
    #noiseSNR = (-10, -5, 0, 5, 10, 15, 20)
    noiseSNR = tuple(range(-5,21))
    #noiseSNR = (20, )
    degs = (10, )
    #degs = (10, )
    lstErrors = []
    errorPlot = go.Figure()



    for deg in degs:
        for snr in noiseSNR:
            print("TESTING with deg", deg, "and snr of", snr, "dB")
            eastPositions, lstToas, cfarData, sigCC, error = underwater_localization(anc, testPath, snr=snr, deg=deg)
            lstErrors.append(np.mean(error))
            
        #errorPlot.add_trace(go.Scatter(x=noiseSNR, y=lstErrors, marker_color='#000000'))
        

    ### plotting positions in 3d
    posScatter = go.Figure()

    # plot anchors themselves
    xAnchors = list(zip(*anc))[0]
    yAnchors = list(zip(*anc))[1]
    zAnchors = list(zip(*anc))[2]

    posScatter.add_trace(go.Scatter3d(x=xAnchors, y=yAnchors, z=zAnchors, mode='markers', marker=dict(size=30), name="Anchors", marker_color='#EF553B', text=["S0", "S1", "S2", "S3"], textposition="bottom center"))
    
    # plot real positions of target
    posScatter.add_trace(go.Scatter3d(x=testPath[0], y=testPath[1], z=testPath[2], name="real Position", marker_color='#636EFA'))

    # plot eastimated position of target
    xEstPos = list(zip(*eastPositions))[0]
    yEstPos = list(zip(*eastPositions))[1]
    zEstPos = list(zip(*eastPositions))[2]
    posScatter.add_trace(go.Scatter3d(x=xEstPos, y=yEstPos, z=zEstPos, name="est. Position", marker_color='#FF97FF'))

    
    scene=dict(camera=dict(up=dict(x=0, y=0, z=2),
                                          center=dict(x=0, y=0, z=-0.1),
                                          eye=dict(x=1.25, y=1.25, z=0.75)))
    posScatter.update_layout(title="Underwater Localization", scene=scene)
    #posScatter.write_image("3dplots/d10snr0.pdf", scale=1, width= 1.6 * 600, height= 1.2 * 600)

    ### plotting correlation toas
    #tdoaLineplot = make_subplots(rows=len(anc))
    tdoaLineplot = go.Figure()

    # plot correlation, cfar
    for ai in range(len(anc)-3):
        tauCC = cfarData[ai][0]
        tauSigCC = cfarData[ai][1]
        tauCFAR = cfarData[ai][2]

        tdoaLineplot.add_trace(go.Scatter(x=tauCC, y=np.abs(tauSigCC), mode='lines', marker_color='#000', name='signal'))
        tdoaLineplot.add_trace(go.Scatter(x=tauCC, y=tauCFAR, mode='lines', marker_color='#636EFA', name='CA-FAR'))

        for t in lstToas[ai]:
            tdoaLineplot.add_vline(t, line_color='#EF553B', line_width=3, line_dash='solid')

    for pos in range(POS):

        selection = [item[pos] for item in lstToas]
        for i in range(len(anc)):
            tdoaLineplot.add_vline(np.min(selection), line_color='#7F7F7F', line_width=3, line_dash='dash')

    tdoaLineplot.update_layout(showlegend=True, xaxis_title='time [s]', yaxis_title='amplitude', yaxis_range=[0,1])
    #tdoaLineplot.write_image("3dplots/d10plane1lines.pdf", scale=2, width= 1.6 * 400, height= 1.8 * 400)
    
    ### show all plots
    #posScatter.show()
    #tdoaLineplot.show()
    
    errorPlot.add_trace(go.Scatter(x=noiseSNR, y=lstErrors, marker_color='#000000'))
    errorPlot.show()
    errorPlot.update_layout(xaxis_title='signal SNR [dB]', yaxis_title='mean distance error [m]')
    errorPlot.write_image("snrerror1b.pdf", scale=1, width= 1.6 * 400, height= 1.2 * 400)
    errorPlot.update_layout(yaxis_range=[0,round(0.25*np.var(lstErrors), 2)])
    errorPlot.write_image("snrerror2b.pdf", scale=1, width= 1.6 * 400, height= 1.2 * 400)

if __name__ == "__main__":
    main()
