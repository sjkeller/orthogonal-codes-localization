from processing import simulation
import plotly.graph_objects as go
import numpy as np
from decimal import Decimal

# source pending
SOFSOUND_WATER = 1500

def totalError(listRealPos: list[tuple], listEstPos: list[tuple]):
    errors = []
    for i in range(len(listRealPos)):
        realPos = np.array(listRealPos[i])
        estPos = np.array(listEstPos[i])
        errors.append(np.linalg.norm(estPos - realPos))

    return errors


"""
def leastSquareEst(anchorPos: list[tuple], delays: list[float]):

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
    dK3 = dX3 ** 2 + dX3 ** 2 + dZ3 ** 2
    dK4 = dX4 ** 2 + dY4 ** 2 + dZ4 ** 2

    matA = np.array([[dX2, dY2, dZ2], [dX3, dY3, dZ3], [dX3, dY4, dZ4]])

    vecC = np.array([dK2 - m12 ** 2, dK3 - m13 ** 2, dK4 - m14 ** 2])
    vecD = np.array([-m12, -m13, -m14])
    vecB = vecC + vecD

    dPos = np.linalg.lstsq(matA, vecB, rcond=None)[0]
    
    return dPos"""

def localLateration(anchorPos: list[tuple], delays: list[float]):

    x0, y0, z0 = anchorPos[0]
    x1, y1, z1 = anchorPos[1]
    x2, y2, z2 = anchorPos[2]
    x3, y3, z3 = anchorPos[3]
    x4, y4, z4 = anchorPos[4]

    tau01 = delays[0]
    tau02 = delays[1]
    tau03 = delays[2]
    tau04 = delays[3]

    d01 = SOFSOUND_WATER * tau01
    d02 = SOFSOUND_WATER * tau02
    d03 = SOFSOUND_WATER * tau03
    d04 = SOFSOUND_WATER * tau04

    matA = np.array(
        [[x0 - x1, y0 - y1, z0 - z1, d01], 
         [x0 - x2, y0 - y2, z0 - z2, d02], 
         [x0 - x3, y0 - y3, z0 - z3, d03], 
         [x0 - x4, y0 - y4, z0 - z4, d04]]
    )
    vecB = np.array(
        [x0 ** 2 - x1 ** 2 + y0 ** 2 - y1 ** 2 + z0 ** 2 - z1 ** 2 + d01 ** 2, 
         x0 ** 2 - x2 ** 2 + y0 ** 2 - y2 ** 2 + z0 ** 2 - z2 ** 2 + d02 ** 2, 
         x0 ** 2 - x3 ** 2 + y0 ** 2 - y3 ** 2 + z0 ** 2 - z3 ** 2 + d03 ** 2,
         x0 ** 2 - x4 ** 2 + y0 ** 2 - y4 ** 2 + z0 ** 2 - z4 ** 2 + d04 ** 2]
    ) / 2

    ### positions by last sqare regression using direct inverse method (A^T * A)^-1 * A^T * b
    # source https://math.stackexchange.com/questions/1722021/trilateration-using-tdoa
    #pos = np.dot(np.dot(np.linalg.inv(np.dot(matA.T, matA)), matA.T), vecB)[:3]
    pos = np.linalg.solve(matA, vecB)
    #pos, resid, _, _ = np.linalg.lstsq(matA, vecB, rcond=None)
    #pos = np.linalg.tensorsolve(matA, vecB)

    #print("residuals", resid)

    return pos

def dist_to_delay(achorPos: list[tuple], vehPos: tuple[float], absolute: bool = False):

    # tau_i = 1/c * (|S - S_0| - |S - S_i|)
    s = np.array(vehPos)
    s0 = np.array(achorPos[0])
    s1 = np.array(achorPos[1])
    s2 = np.array(achorPos[2])
    s3 = np.array(achorPos[3])
    s4 = np.array(achorPos[4])

    dist0 = np.linalg.norm(s - s0)
    dist1 = np.linalg.norm(s - s1)
    dist2 = np.linalg.norm(s - s2)
    dist3 = np.linalg.norm(s - s3)
    dist4 = np.linalg.norm(s - s4)

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
        t4 = dist4 / SOFSOUND_WATER

        return t0, t1, t2, t3, t4

    else:
        tau01 = (dist0 - dist1) / SOFSOUND_WATER
        tau02 = (dist0 - dist2) / SOFSOUND_WATER
        tau03 = (dist0 - dist3) / SOFSOUND_WATER
        tau04 = (dist0 - dist4) / SOFSOUND_WATER

        return tau01, tau02, tau03, tau04

def circlePath(scale: float, offset: float, res: int = 10):
    phi = np.linspace(0, 4*np.pi, res, endpoint=False)
    x = scale * np.sin(phi) + offset
    y = scale * np.cos(phi) + offset
    #z = np.linspace(0, -res, res)
    z = np.array(range(0, res))

    return x,y,z

def main():
    global file_index, load_index
    #lstSNRdB = [-15, -10, -5, 0, 5]
    lstSNRdB = [5]
    #chns = ['PLANE1', 'PLANE2', 'CASTLE1', 'CASTLE2', 'CASTLE3']
    degs = [8, 9, 10]
    

    #anc = [(np.random.randint(0,100), np.random.randint(0,100),np.random.randint(-50,0)), 
    #    (np.random.randint(0,100), np.random.randint(0,100),np.random.randint(-50,0)), 
    #    (np.random.randint(0,100), np.random.randint(0,100),np.random.randint(-50,0)), 
    #    (np.random.randint(0,100), np.random.randint(0,100), np.random.randint(-50,0))]

    #anc = [(10,1,np.random.randint(5,20)), (100,10,np.random.randint(5,20)), (10,100,np.random.randint(5,20)), (100,100,np.random.randint(5,20)), (50,50,np.random.randint(5,20))]
    anc = [(10,1,10),(100,10,15),(100,100,20),(10,100,5), (10,50,10)]
    #print("Real delays", realDelays)

    lstRealPos = []
    lstEstPos = []

    xPos, yPos, zPos = circlePath(15, 75, 15)

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

    fig.add_trace(go.Scatter3d(x=xAnchors, y=yAnchors, z=zAnchors, mode='markers', marker=dict(size=30), name="Anchors", marker_color='#EF553B', text=["S0", "S1", "S2", "S3", "S4"], textposition="bottom center"))
    snr = -10
    # sweet spot -11dB, everyting above creates intense distortion
    snr = 10 ** (snr / 10)
    for cx, cy, cz in zip(xPos, yPos, zPos):

        relDelays = dist_to_delay(anc, (cx, cy, cz))
        absDelays = dist_to_delay(anc, (cx, cy, cz), absolute=True)

        lstRealPos.append((cx, cy, cz))
        
        maxDelays, cfarDelays = simulation(absDelays, 5, showAll=False, addGWN=True, useSim=True, channel='CASTLE1', polyDeg=10, targetSNR=snr)

        #print("rel delays from direct calc", round(relDelays[0],3), round(relDelays[1],3), round(relDelays[2],3))
        #print("rel. delays from simulation", round(maxDelays[0],3), round(maxDelays[1],3), round(maxDelays[2],3))

        estPos = localLateration(anc, maxDelays)
        lstEstPos.append((estPos[0], estPos[1], estPos[2]))

    fig.add_trace(go.Scatter3d(x=xPos, y=yPos, z=zPos, name="real Position", marker_color='#636EFA'))

    xEstPos = list(zip(*lstEstPos))[0]
    yEstPos = list(zip(*lstEstPos))[1]
    zEstPos = list(zip(*lstEstPos))[2]

    fig.add_trace(go.Scatter3d(x=xEstPos, y=yEstPos, z=zEstPos, name="est. Position", marker_color='#AB63FA'))

    fig.show()


    fig = go.Figure()

    localError = totalError(lstRealPos, lstEstPos)

    fig.add_trace(go.Scatter(y=localError))
    fig.show()
    # add filtfilt order dobuling to PDF

if __name__ == "__main__":
    main()
    