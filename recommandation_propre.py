import pandas as pd
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF

class Recommandation(object):

    def __init__(self, database):
        self.database = database

    def rec_svd(self, user, film):
        mtx = sp.csr_matrix(self.database.replace(np.nan, 0).values)
        svd = TruncatedSVD(n_components=400)
        svd.fit(mtx)
        q = svd.transform(mtx)
        df_q = pd.DataFrame(q, index=self.database.index)
        p = svd.components_
        df_p = pd.DataFrame(p, columns=self.database.columns)
        return((df_q.loc[film]*df_p[user]).sum())

    def rec_nfm(self, user, film):
        mtx = sp.csr_matrix(self.database.replace(np.nan, 0).values)
        svd = NMF(n_components=400, init='random', random_state=0)
        svd.fit(mtx)
        q = svd.transform(mtx)
        df_q = pd.DataFrame(q, index=self.database.index)
        p = svd.components_
        df_p = pd.DataFrame(p, columns=self.database.columns)
        return((df_q.loc[film]*df_p[user]).sum())
