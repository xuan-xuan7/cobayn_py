import numpy as np

def getBestSet(bench, selectionSize, ALO=None, DLO=None):
    row = len(bench)
    col = len(bench[0])
    if ALO is None and DLO is None:
        ALO = []
        DLO = []
    elif ALO is not None:
        DLO = np.arange(col)

    bench_size = row * col
    bestSet = np.zeros(((row - len(ALO)) * (col - len(DLO)) * selectionSize, len(bench[0][0]['micaNorm']) + bench[0][0]['DB'].shape[1]))
    ExT = np.zeros((bench_size * selectionSize, 1))

    arrayPointer = 0
    for a in range(row):
        for d in range(col):
            if (a + 1) not in ALO or (d + 1) not in DLO:
                myExT = bench[a][d]['Y']
                sortedT = np.sort(myExT)
                ids = np.argsort(myExT)

                myBestDB = bench[a][d]['DB'][ids[:selectionSize], :]

                bestSet[arrayPointer:(arrayPointer + selectionSize), 0: len(bench[a][d]['micaNorm'])] = np.tile(bench[a][d]['micaNorm'], (selectionSize, 1))
                bestSet[arrayPointer:(arrayPointer + selectionSize), len(bench[a][d]['micaNorm']): len(bench[a][d]['micaNorm']) + bench[a][d]['DB'].shape[1]] = myBestDB

                ExT[arrayPointer:(arrayPointer + selectionSize)] = sortedT[:selectionSize].reshape(25, 1)

                arrayPointer += selectionSize

    return bestSet, ExT
