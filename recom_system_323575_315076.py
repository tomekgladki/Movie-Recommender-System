import argparse
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF as NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDRegressor

def parse_arguments():
    parser = argparse.ArgumentParser(description = 'Movie Recommender System')
    parser.add_argument('--train', help = 'train file', default = 'ratings_train.csv', required=False)
    parser.add_argument('--test', help = 'test file', default = 'ratings_test.csv', required=False)
    parser.add_argument('--alg', help = 'Method (default: NMF, others: SVD1, SVD2, SGD)', default = 'SGD')
    parser.add_argument('--result', help = 'file where final RMSE will be saved', default = 'result.csv')
    args = parser.parse_args()

    return args.train, args.test, args.alg, args.result

train, test, alg, result = parse_arguments()


train_df = pd.read_csv(train)
test_df = pd.read_csv(test)

train_df = train_df.pivot_table(index="userId", columns="movieId", values="rating")
test_df = test_df.pivot_table(index="userId", columns="movieId", values="rating")
#df_3 = df.copy(deep=True).fillna(3)

def fillNanByColRowMeans(DataFrame, alpha=0.1):
    dfc = alpha * DataFrame.copy(deep=True).apply(lambda c: c.fillna(c.mean()), axis=0)
    dfr = (1-alpha) * DataFrame.copy(deep=True).apply(lambda c: c.fillna(c.mean()), axis=1)
    ResultDF = dfc.add(dfr, fill_value=0).copy(deep=True)

    return ResultDF
#df_cr = fillNanByColRowMeans(DataFrame=test_df).copy(deep=True)
def RMSE(Z_prim, T):
    dimZ = Z_prim.shape
    dimT =T.shape
    concat = pd.concat([T, Z_prim], ignore_index=False)
    newT = concat[:dimT[0]]
    newZ = concat[(dimT[0]):(dimT[0]+dimZ[0])]
    sub2 = newZ.sub(newT)**2
    mean = sub2.mean(skipna=True).mean()

    return mean**0.5

def aprox_SGD(DataFrame, testDF, epchs=10, n_fctrs=13, Lambda=9, gamma=0.01):
    n_u = DataFrame.shape[0]
    n_m = DataFrame.shape[1]
    #print(n_u)

    W = pd.DataFrame(np.random.rand(n_u, n_fctrs)) # 610 x n_fctrs
    #print(W)
    H = pd.DataFrame(np.random.rand(n_m, n_fctrs))  # 9386 x n_fctrs
    #print(H)
    for e in range(epchs):
        #print(e)
        for tple in zip(DataFrame.index, DataFrame.columns):
            gamma = gamma / (1 + 0.01 * e) #DECAY
            r = tple[0]-1
            c = tple[1]
            d = DataFrame.iloc[r, c] - (W.iloc[r, :] @ H.iloc[c, :].T)
            #print(d)
            if r==n_u or c==n_m: break
            W.iloc[r+1, :] += -gamma * (
                        np.dot(-2 * H.iloc[c, :], d * np.dot(W.iloc[r, :], H.iloc[c, :])) + 2 * Lambda * W.iloc[r, :])
            H.iloc[c+1, :] += -gamma * (
                        np.dot(-2 * W.iloc[r, :], d * np.dot(W.iloc[r, :], H.iloc[c, :])) + 2 * Lambda * H.iloc[c, :])
        Z_aprox = W @ H.transpose()
        Z_aprox = pd.DataFrame(Z_aprox,
                               columns=list(DataFrame.columns),
                               index=list(DataFrame.index))
        rmse = RMSE(Z_aprox, testDF)
        #print(rmse)
        if rmse < 0.8:
            return Z_aprox
    #print(Z_aprox)
    return Z_aprox
def aprox_NMF(Z, r=15):
    model = NMF(n_components=r, init="random",
                random_state=0, max_iter=200)
    W = model.fit_transform(Z)
    H = model.components_
    Z_aprox = np.dot(W,H)
    Z_aprox = pd.DataFrame(Z_aprox,
                           columns=list(Z.columns),
                           index=list(Z.index))
    return Z_aprox
def aprox_SVD(Z, r=10):
    model = TruncatedSVD(n_components=r, random_state=42)
    model.fit(Z)
    Sigma2 = np.diag(model.singular_values_)
    VT = model.components_
    W = model.transform(Z) / model.singular_values_
    H = np.dot(Sigma2, VT)
    Z_aprox = np.dot(W, H)
    Z_aprox = pd.DataFrame(Z_aprox,
                           columns=list(Z.columns),
                           index=list(Z.index))
    return Z_aprox
def aprox_SVD2(Z, testDF, isNanDF, iter=40, r=2, stop_condit=0):
    i = iter-1
    Z_aprox = aprox_SVD(Z, r)
    # tam gdzie w testowym jes Nan ma byc aprox,
    # a tam gdzie notNan ma byc rzeczywista wart !
    Z_aprox = Z_aprox * (1*isNanDF) + Z * (isNanDF*(-1)+1)
    Z_aprox = pd.DataFrame(Z_aprox,
                           columns=Z.columns,
                           index=Z.index)
    #print( Z_aprox)
    if stop_condit > 0:
        rmse = RMSE(Z_aprox, testDF)
        if rmse < stop_condit:
            return Z_aprox
    if i == 0:
        return Z_aprox
    return aprox_SVD2(Z_aprox, testDF, isNanDF, iter=i, r=r, stop_condit=stop_condit)

def function(ALG=alg):
    if ALG == "NMF":
        tr_df = fillNanByColRowMeans(train_df).copy(deep=True)
        rmse = RMSE(
            aprox_NMF(tr_df),
            test_df)
    if ALG == "SVD1":
        tr_df = fillNanByColRowMeans(train_df).copy(deep=True)
        rmse = RMSE(
            aprox_SVD(tr_df),
            test_df)
    if ALG == "SVD2":
        NanCoordinates = train_df.isna()
        tr_df = fillNanByColRowMeans(train_df, alpha=0.1).copy(deep=True)
        rmse = RMSE(
            aprox_SVD2(Z=tr_df, testDF=test_df, isNanDF=NanCoordinates),
            test_df)
    if ALG == "SGD":
        tr_df = fillNanByColRowMeans(train_df).copy(deep=True)
        rmse = RMSE(
            aprox_SGD(tr_df, test_df),
            test_df)
    return rmse

if __name__ == '__main__':
    res = pd.DataFrame({'RMSE': [function(alg)]})
    #print(res)
    res.to_csv(result, index=False)

