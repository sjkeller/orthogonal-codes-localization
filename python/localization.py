from processing import simulation

def main():
    global file_index, load_index
    #lstSNRdB = [-15, -10, -5, 0, 5]
    lstSNRdB = [5]
    chns = ['PLANE1', 'PLANE2', 'CASTLE1', 'CASTLE2', 'CASTLE3']
    degs = [8, 9, 10]
    for snr in lstSNRdB:
        snr = 10 ** (snr / 10)
        delays = simulation([10e-3, 20e-3, 30e-3], 3, showAll=False, addGWN=True, targetSNR=snr, useSim=True, channel='PLANE1', polyDeg=8)
        print(delays)
        file_index = 0
        load_index = 0

    #showButterBode()
    #showRaisedCosine()


if __name__ == "__main__":
    main()
    