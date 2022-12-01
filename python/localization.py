from processing import simulation
import plotly.graph_objects as go
import numpy as np

# source pending
SOFSOUND_WATER = 1500

def leastSquareEst(anchorPos: list[tuple], delays: list[float]):

    #m12 = (delays[1] - delays[0]) * SOFSOUND_WATER
    #m13 = (delays[2] - delays[0]) * SOFSOUND_WATER
    #m14 = (delays[3] - delays[0]) * SOFSOUND_WATER

    m12 = delays[0] * SOFSOUND_WATER
    m13 = delays[0] * SOFSOUND_WATER
    m14 = delays[0] * SOFSOUND_WATER

    dX2 = anchorPos[1][0] - anchorPos[0][0]
    dX3 = anchorPos[2][0] - anchorPos[0][0]
    dX4 = anchorPos[3][0] - anchorPos[0][0]

    dY2 = anchorPos[1][1] - anchorPos[0][1]
    dY3 = anchorPos[2][1] - anchorPos[0][1]
    dY4 = anchorPos[3][1] - anchorPos[0][1]

    dZ2 = anchorPos[1][2] - anchorPos[0][2]
    dZ3 = anchorPos[2][2] - anchorPos[0][2]
    dZ4 = anchorPos[3][2] - anchorPos[0][2]

    dK2 = dX2 ** 2 + dY2 ** 2 + dZ2 ** 2
    dK3 = dX3 ** 2 + dX3 *+ 2 + dZ3 ** 2
    dK4 = dX4 ** 2 + dY4 ** 2 + dZ4 ** 2

    matA = np.array([[dX2, dY2, dZ2], [dX3, dY3, dZ3], [dX3, dY4, dZ4]])

    vecC = np.array([dK2 - m12 ** 2, dK3 - m13 ** 2, dK4 - m14 ** 2])
    vecD = np.array([-m12, -m13, -m14])
    vecB = vecC + vecD

    dPos = np.linalg.lstsq(matA, vecB, rcond=None)[0]
    
    return dPos

def localLateration(anchorPos: list[tuple], delays: list[float]):

    x0, y0, z0 = anchorPos[0]
    x1, y1, z1 = anchorPos[1]
    x2, y2, z2 = anchorPos[2]
    x3, y3, z3 = anchorPos[3]

    tau01 = delays[0]
    tau02 = delays[1]
    tau03 = delays[2]

    d01 = SOFSOUND_WATER * tau01
    d02 = SOFSOUND_WATER * tau02
    d03 = SOFSOUND_WATER * tau03

    matA = np.array([[x0 - x1, y0 - y1, z0 - z1, d01], [x0 - x2, y0 - y2, z0 - z2, d02], [x0 - x3, y0 - y3, z0 - z3, d03]])
    vecB = np.array(
        [x0 ** 2 - x1 ** 2 + y0 ** 2 - y1 ** 2 + z0 ** 2 - z1 ** 2 + d01 ** 2, 
         x0 ** 2 - x2 ** 2 + y0 ** 2 - y2 ** 2 + z0 ** 2 - z2 ** 2 + d02 ** 2, 
         x0 ** 2 - x3 ** 2 + y0 ** 2 - y3 ** 2 + z0 ** 2 - z3 ** 2 + d03 ** 2]
    ) / 2

    ### positions by last sqare regression using direct inverse method (A^T * A)^-1 * A^T * b
    # source https://math.stackexchange.com/questions/1722021/trilateration-using-tdoa
    #pos = np.dot(np.dot(np.linalg.inv(np.dot(matA.T, matA)), matA.T), vecB)[:2]
    pos = np.linalg.lstsq(matA, vecB, rcond=None)[0]

    return pos

def dist_to_delay(achorPos: list[tuple], vehPos: tuple[float], absolute: bool = False):

    # tau_i = 1/c * (|S - S_0| - |S - S_i|)
    s = np.array(vehPos)
    s0 = np.array(achorPos[0])
    s1 = np.array(achorPos[1])
    s2 = np.array(achorPos[2])
    s3 = np.array(achorPos[3])

    dist0 = np.linalg.norm(s - s0)
    dist1 = np.linalg.norm(s - s1)
    dist2 = np.linalg.norm(s - s2)
    dist3 = np.linalg.norm(s - s3)

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

        return t0, t1, t2, t3

    else:
        tau01 = (dist0 - dist1) / SOFSOUND_WATER
        tau02 = (dist0 - dist2) / SOFSOUND_WATER
        tau03 = (dist0 - dist3) / SOFSOUND_WATER

        return tau01, tau02, tau03

def circlePath(scale: float, offset: float, res: int = 10):
    phi = np.linspace(0, 2*np.pi, res)
    x = scale * np.sin(phi) + offset
    y = scale * np.cos(phi) + offset
    #z = np.linspace(0, -res, res)
    z = np.zeros(res) + 1

    return x,y,z

def main():
    global file_index, load_index
    #lstSNRdB = [-15, -10, -5, 0, 5]
    lstSNRdB = [5]
    #chns = ['PLANE1', 'PLANE2', 'CASTLE1', 'CASTLE2', 'CASTLE3']
    degs = [8, 9, 10]
    snr = 0

    anc = [(0,0,0), (0,100,0), (100,0,0), (100, 100, 0)]
    
    #print("Real delays", realDelays)

    lstRealDelays = []
    lstEstPos = []

    xPos, yPos, zPos = circlePath(15, 25, 25)

    """for ch in chns:
        snr = 10 ** (snr / 10)
        maxDelays, cfarDelays = simulation(realDelays, 3, showAll=False, addGWN=False, targetSNR=snr, useSim=True, channel=ch, polyDeg=10)
        print("Detected delays by max:", maxDelays)
        print("Detected delays by cfar:", cfarDelays)
        file_index = 0
        load_index = 0

        estPos = localLateration(anc, realDelays)
        print("new pos", estPos)"""

    #showButterBode()
    #showRaisedCosine()

    fig = go.Figure()

    xAnchors = list(zip(*anc))[0]
    yAnchors = list(zip(*anc))[1]
    zAnchors = list(zip(*anc))[2]

    fig.add_trace(go.Scatter3d(x=xAnchors, y=yAnchors, z=zAnchors, mode='markers', marker=dict(size=30), name="Anchors", marker_color='#EF553B', text=["S0", "S1", "S2"], textposition="bottom center"))
    #fig.add_trace(go.Scatter(x=[realPos[0]], y=[realPos[1]], mode='markers', marker=dict(size=50), name="real Position", marker_color='#636EFA'))
    #fig.add_trace(go.Scatter(x=[estPos[0]], y=[estPos[1]], mode='markers', marker=dict(size=50), name="est. Position", marker_color='#AB63FA'))

    for cx, cy, cz in zip(xPos, yPos, zPos):

        relDelays = dist_to_delay(anc, (cx, cy, cz))
        absDelays = dist_to_delay(anc, (cx, cy, cz), absolute=True)

        lstRealDelays.append((cx, cy, cz))
        snr = 10 ** (snr / 10)
        maxDelays, cfarDelays = simulation(absDelays, 4, showAll=False, addGWN=False, targetSNR=snr, useSim=True, channel='CASTLE1', polyDeg=10)

        print("rel delays from direct calc", round(relDelays[0],3), round(relDelays[1],3), round(relDelays[2],3))
        print("rel. delays from simulation", round(maxDelays[0],3), round(maxDelays[1],3), round(maxDelays[2],3))

        estPos = localLateration(anc, maxDelays)
        print("est. Pos", estPos)
        lstEstPos.append((estPos[0], estPos[1], estPos[2]))

    fig.add_trace(go.Scatter3d(x=xPos, y=yPos, z=zPos, name="real Position", marker_color='#636EFA'))

    xEstPos = list(zip(*lstEstPos))[0]
    yEstPos = list(zip(*lstEstPos))[1]
    zEstPos = list(zip(*lstEstPos))[2]

    fig.add_trace(go.Scatter3d(x=xEstPos, y=yEstPos, z=zEstPos, name="est. Position", marker_color='#AB63FA'))

    fig.show()

    # add filtfilt order dobuling to PDF



if __name__ == "__main__":
    main()
    