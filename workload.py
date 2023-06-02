import os
import sys
from custom_emissions_tracker import EmissionsTracker
from sklearn import metrics
from sklearn.linear_model import SGDOneClassSVM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from custom_emissions_tracker import EmissionsTracker

from dotenv import load_dotenv

load_dotenv()


class Workload:
    def __init__(self, measure_power_secs):
        self.tracker = EmissionsTracker(measure_power_secs=measure_power_secs, tracking_mode='process')
        try:
            print('THIS IS ENV ', os.getenv('DATASET_CSV'))

        except Exception as ex:
            print("EXCEPTION, I TRIED TO READ")
        self.df = pd.read_csv(os.getenv('DATASET_CSV'))

    def isolation_forrest(self, n_estimators=20, max_samples=100, contamination=0.01, random_state=42, verbose=2):
        print('-ISOLATION FORREST-')

        self.df['Time'] = self.df['Time'].apply(lambda x: x / 3600)
        df_norm = self.df.copy()
        df_norm['Time'] = StandardScaler().fit_transform(df_norm['Time'].values.reshape(-1, 1))
        df_norm['Amount'] = StandardScaler().fit_transform(df_norm['Amount'].values.reshape(-1, 1))
        train, test = train_test_split(df_norm, test_size=0.3, random_state=10)
        X_train = train[train['Class'] == 0]
        X_train = X_train.drop(['Class'], axis=1)

        X_test = test.drop(['Class'], axis=1)
        y_test = test['Class']
        model_iF = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                                   contamination=contamination, random_state=random_state, verbose=verbose)
        model_iF.fit(X_train)
        y_pred = model_iF.predict(X_test)
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1

        f1_score(y_test, y_pred)

        return metrics.roc_auc_score(y_test, y_pred)

    def grid_search_isolation_forrest(self):
        print("GRID-SEARCH-ISOLATION FORREST IS STARTING")
        estimators = [20, 40, 60]
        max_samples = [100, 200, 400]
        contamination = [0.1]#[0.1, 0.01, 0.001]
        random_states = [42]#[42, 62, 82]
        verbose = [8]#[8, 16, 32]
        auc_score = []
        for est in estimators:
            for samps in max_samples:
                for cont in contamination:
                    for ran_state in random_states:
                        for ver in verbose:
                            auc_score.append({
                                "n_estimators": est,
                                "max_samples": samps,
                                "contamination": cont,
                                "random_state": ran_state,
                                "verbose": ver,
                                "auc": self.isolation_forrest(est, samps, cont, ran_state, ver)
                            }

                            )
        best_auc = sorted(auc_score, key=lambda d: d['auc'])[-1]
        return best_auc

    def autoencoder(self, learning_rate=0.00001, first_layer=16, second_layer=14, third_layer=12, fourth_layer=10,
                    fifth_layer=8, sixth_layer=6, seventh_layer=4, patience=5, verbose=1):
        print('-AUTOENCODER-')
        df_sc = self.df.copy()
        df_sc['Time'] = StandardScaler().fit_transform(df_sc['Time'].values.reshape(-1, 1))
        df_sc['Amount'] = StandardScaler().fit_transform(df_sc['Amount'].values.reshape(-1, 1))
        train, test = train_test_split(df_sc, test_size=0.3, random_state=10)
        X_train = train[train['Class'] == 0]
        X_train = X_train.drop(['Class'], axis=1)
        learning_rate = learning_rate
        input_dim = X_train.shape[1]
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(first_layer, activation='elu', activity_regularizer=regularizers.l1(learning_rate))(input_layer)
        encoder = Dense(second_layer, activation='relu')(encoder)
        encoder = Dense(third_layer, activation='relu')(encoder)
        encoder = Dense(fourth_layer, activation='relu')(encoder)
        encoder = Dense(fifth_layer, activation='relu')(encoder)
        encoder = Dense(sixth_layer, activation='relu')(encoder)
        encoder = Dense(seventh_layer, activation='relu')(encoder)
        encoder = Dense(sixth_layer, activation='relu')(encoder)
        encoder = Dense(fifth_layer, activation='relu')(encoder)
        encoder = Dense(fourth_layer, activation='relu')(encoder)
        encoder = Dense(third_layer, activation='relu')(encoder)
        decoder = Dense(second_layer, activation='relu')(encoder)
        decoder = Dense(first_layer, activation='relu')(decoder)
        decoder = Dense(input_dim, activation='elu')(decoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam',
                            metrics=['accuracy'],
                            loss='mean_squared_error')
        EarlyStopping(monitor='accuracy', patience=patience, verbose=verbose)
        autoencoder.summary()

    def svm(self, random_state=42, nu=0.1, tol=1e-4):
        print("-SVM-")
        df_sc = self.df.copy()
        df_sc['Time'] = StandardScaler().fit_transform(df_sc['Time'].values.reshape(-1, 1))
        train, test = train_test_split(df_sc, test_size=0.3, random_state=10)
        X_train = train[train['Class'] == 0]
        X_train = X_train.drop(['Class'], axis=1)
        X_test = test.drop(['Class'], axis=1)
        model_sgd = SGDOneClassSVM(random_state=random_state, nu=nu, fit_intercept=True, shuffle=True, tol=tol)
        model_sgd.fit(X_train)
        pred = model_sgd.predict(X_test)
        pred[pred == 1] = 0
        pred[pred == -1] = 1

    def grid_search_svm(self):
        random_state = [42, 142, 1420, 14200, 142000]
        nu = [0.1, 0.01, 0.001, 0.0001, 0.0001]
        tol = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        for rand_state in random_state:
            for n in nu:
                for t in tol:
                    self.svm(random_state=rand_state, nu=n, tol=t)

    def grid_search_autoencoder(self):
        learing_rates = [0.1, 0.001, 0.0001, 0.00001]
        layers = [
            [2048, 1024, 512, 256, 128, 64, 32],
            [4056, 2048, 1024, 512, 256, 128, 64],
            [8162, 4056, 2048, 1024, 512, 256, 128],
            [16224, 8162, 4056, 2048, 1024, 512, 256]]
        verb = [1, 2, 10, 100]
        pat = [5, 25, 50, 500]
        for learn in learing_rates:

            for layer in layers:
                for v in verb:
                    for p in pat:
                        print(f'params: layer {layer} verbose {v} patience {p} learning {learn}')
                        self.autoencoder(learn, *layer, patience=p, verbose=v)

    def compute_workload_consumption(self, workload: str):
        try:

            if os.path.exists(f"{os.getenv('CONSUMPTION_DIR')}/{workload}_consumption.csv"):
                os.remove(f"{os.getenv('CONSUMPTION_DIR')}/{workload}_consumption.csv")
            if os.path.exists(f"{os.getenv('CONSUMPTION_DIR')}/Custom_consumption.csv"):
                os.remove(f"{os.getenv('CONSUMPTION_DIR')}/Custom_consumption.csv")

            print('-Start tracking energy consumption-')
            self.tracker.start()
            if workload == 'isolation_forrest':
                self.grid_search_isolation_forrest()
            if workload == 'svm':
                self.grid_search_svm()
            if workload == 'autoencoder':
                self.grid_search_autoencoder()

            self.tracker.stop()
            os.rename(f"{os.getenv('CONSUMPTION_DIR')}/Custom_Consumption.csv",
                      f"{os.getenv('CONSUMPTION_DIR')}/{workload}_consumption.csv")
        except Exception as ex:
            print(ex)
