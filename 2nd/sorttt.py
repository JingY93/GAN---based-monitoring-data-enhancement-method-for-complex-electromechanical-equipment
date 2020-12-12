import numpy as np
# def sorttt(matrix_biaozhun,matrix_chuli):
#     m,n=matrix_biaozhun.shape()
#     SORT=np.zeros(m,n)
#     for i in range(m):
#         SORT[i,:]=np.argsort(np.argsort(matrix_biaozhun[i,:]))
#     for j in range(m):
#         matrix_chuli=sorted(matrix_chuli)
#     pass
def sortt(matrix_biaozhun,matrix_chuli):
    m=matrix_biaozhun.shape[0]
    SORT=np.argsort(np.argsort(matrix_biaozhun))
    matrix_chuli=sorted(matrix_chuli)
    matrix_chuli = np.array(matrix_chuli)
    new=[]
    for i in range(m):
        new.append(matrix_chuli.take(SORT[i]))
    new = np.array(new)
    return new