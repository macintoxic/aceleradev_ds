import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE



def main():
    print("iniciando")
    df = pd.read_csv('train.csv')

    print("Dados do data frame")
    df.head()

    df=df[[c for c in df.columns if not c.startswith('IN_')]]
    df=df[[c for c in df.columns if not c.startswith('CO_')]]
    df=df[[c for c in df.columns if not c.startswith('TX_')]]
    df=df[[c for c in df.columns if not c.startswith('TP_')]]

    df = df.drop('Unnamed: 0', axis=1)
    df = df.drop('NU_ANO', axis=1)
    df = df.drop('NU_IDADE', axis=1)


    df = df.select_dtypes(np.number)

    df.head()
    df.info()

    print(list(df.columns))
    # pega a matriz de correlação
    # corrmat = df.corr()
    # cols = corrmat.nlargest(4, 'NU_NOTA_MT')['NU_NOTA_MT'].index

    cols = use_correlation(df, 8)
    #cols = use_rfe(df)

    # adiciona a lista
    cols = list(cols)
    # cols.append('NU_INSCRICAO')
    print("Colunas relevantes selecionadas:", cols)
#     df = df[df.NU_NOTA_MT > 0]
#     df = df[df.TP_STATUS_REDACAO == 1]
#     df = df[df.TP_PRESENCA_MT == 1]

    # monta um novo dataframe somente com as colunas selecionadas
    # removendo a coluna NU_NOTA_MT,
    data = df[cols]
    data.fillna(0, inplace=True)

    print(data.head())


    X = data.drop('NU_NOTA_MT', axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, data.NU_NOTA_MT, )

    lm = LinearRegression()
    lm.fit(X_train, Y_train, )

    data_test = pd.read_csv('test.csv')
    result = pd.DataFrame()
    result.insert(0, 'NU_INSCRICAO', data_test.NU_INSCRICAO, True)


    data_test = data_test[cols[1:]]

    data_test.fillna(0, inplace=True)


    result.insert(1, 'NU_NOTA_MT', lm.predict(data_test), True)

    result = result[result.NU_NOTA_MT > 0]

    result.to_csv('answer.csv', index=False, header=True)


def use_correlation(df, features):
    corrmat = df.corr()
    cols = corrmat.nlargest(features, 'NU_NOTA_MT')['NU_NOTA_MT'].index

    # adiciona a lista
    cols = list(cols)
    # print(cols)
    return cols

def use_rfe(df):
    tmp = df.select_dtypes(np.number)
    tmp.fillna(0, inplace=True)
    tmp.head()
    X = tmp.drop(columns='NU_NOTA_MT')
    y = tmp.NU_NOTA_MT
    selector = RFE(LinearRegression(),  step=1, verbose=1, n_features_to_select=5).fit(X, y)

    cols = list(X.loc[:, selector.support_].columns)
    cols.insert(0 , 'NU_NOTA_MT')
    # print(cols)
    return cols




if __name__ == "__main__":
    main()


