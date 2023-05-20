import os

from matplotlib import pyplot as plt
from sklearn.linear_model import SGDOneClassSVM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from dotenv import load_dotenv
load_dotenv('.env')
#df = pd.read_csv('csv_dir/creditcard.csv')

df = pd.read_csv(os.getenv('DATASET_CSV'))





def isolation_forrest():
    print('-ISOLATION FORREST-')

    df['Time'] = df['Time'].apply(lambda x: x / 3600)
    df_norm = df.copy()
    df_norm['Time'] = StandardScaler().fit_transform(df_norm['Time'].values.reshape(-1, 1))
    df_norm['Amount'] = StandardScaler().fit_transform(df_norm['Amount'].values.reshape(-1, 1))
    train, test = train_test_split(df_norm, test_size=0.3, random_state=10)
    X_train = train[train['Class'] == 0]
    X_train = X_train.drop(['Class'], axis=1)

    X_test = test.drop(['Class'], axis=1)
    y_test = test['Class']
    model_iF = IsolationForest(n_estimators=20, max_samples=100,
                               contamination=0.01, random_state=42, verbose=2)
    model_iF.fit(X_train)
    y_pred = model_iF.predict(X_test)
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    f1_score(y_test, y_pred)
    confusion=confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion)
    disp.plot()
    plt.show()
    print(confusion)
    # corr = df.corr()
    #
    # plt.figure(figsize=(20, 8))
    # ax = sns.heatmap(corr.round(2), annot=True, linewidth=0.5, fmt='0.1f', cmap='coolwarm')
    # ax.set_ylim(sorted(ax.get_xlim(), reverse=True))
    # ax.set(title="Correlation Matrix");
    # plt.show()

    metrics = classification_report(y_test, y_pred, output_dict=True)
    return metrics



def local_outlier_factor():
    print('-LOCAL OUTLIER FACTOR-')

    df_norm = df.copy()
    train, test = train_test_split(df_norm, test_size=0.3, random_state=10)
    X_test = test.drop(['Class'], axis=1)
    train, test = train_test_split(df_norm, test_size=0.3, random_state=10)
    X_train = train[train['Class'] == 0]
    X_train = X_train.drop(['Class'], axis=1)
    model_lf = LocalOutlierFactor(n_neighbors=2, contamination=0.1)
    model_lf.fit(X_train)
    model_lf.fit_predict(X_test)
    model_lf.negative_outlier_factor_


def autoencoder():
    print('-AUTOENCODER-')
    df_sc = df.copy()
    df_sc['Time'] = StandardScaler().fit_transform(df_sc['Time'].values.reshape(-1, 1))
    df_sc['Amount'] = StandardScaler().fit_transform(df_sc['Amount'].values.reshape(-1, 1))
    train, test = train_test_split(df_sc, test_size=0.3, random_state=10)
    X_train = train[train['Class'] == 0]
    X_train = X_train.drop(['Class'], axis=1)
    learning_rate = 0.00001
    input_dim = X_train.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(16, activation='elu', activity_regularizer=regularizers.l1(learning_rate))(input_layer)
    encoder = Dense(8, activation='relu')(encoder)
    encoder = Dense(4, activation='relu')(encoder)
    decoder = Dense(8, activation='relu')(encoder)
    decoder = Dense(16, activation='relu')(decoder)
    decoder = Dense(input_dim, activation='elu')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam',
                        metrics=['accuracy'],
                        loss='mean_squared_error')
    EarlyStopping(monitor='accuracy', patience=5, verbose=1)
    autoencoder.summary()


def svm():
    print("-SVM-")
    df_sc = df.copy()
    df_sc['Time'] = StandardScaler().fit_transform(df_sc['Time'].values.reshape(-1, 1))
    df_sc['Time'] = StandardScaler().fit_transform(df_sc['Time'].values.reshape(-1, 1))
    train, test = train_test_split(df_sc, test_size=0.3, random_state=10)
    X_train = train[train['Class'] == 0]

    X_train = X_train.drop(['Class'], axis=1)
    X_test = test.drop(['Class'], axis=1)
    model_sgd = SGDOneClassSVM(random_state=42, nu=0.1, fit_intercept=True, shuffle=True, tol=1e-4)
    model_sgd.fit(X_train)
    pred = model_sgd.predict(X_test)
    pred[pred == 1] = 0
    pred[pred == -1] = 1


def total_workload():
    for i in range(0, 300):
        isolation_forrest()
        svm()
        autoencoder()

#TODO GRID SEARCH -> GET AUC SCORE


if __name__ == '__main__':
    total_workload()
