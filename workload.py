import os
import subprocess
import sys
from custom_emissions_tracker import EmissionsTracker
from sklearn import metrics
import tensorflow as tf
from sklearn.linear_model import SGDOneClassSVM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score
from codecarbon.external.hardware import GPU


from custom_emissions_tracker import EmissionsTracker
import subprocess
from dotenv import load_dotenv

load_dotenv()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


class Workload:
    def __init__(self):

        try:
            print('THIS IS ENV ', os.getenv('DATASET_CSV'))

        except Exception as ex:
            print("EXCEPTION, I TRIED TO READ")
        self.df = pd.read_csv(os.getenv('DATASET_CSV'))

    # TODO try to implet here
    def hf_sca(self):
        pass

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
        contamination = [0.1, 0.01, 0.001]
        random_states = [42, 62, 82]
        verbose = [8, 16, 32]
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

    def autoencoder(self, learning_rate=0.00001, batch_size=64, act='relu'):
        print('-AUTOENCODER-')
        df_sc = self.df.copy()
        df_sc['Time'] = StandardScaler().fit_transform(df_sc['Time'].values.reshape(-1, 1))
        df_sc['Amount'] = StandardScaler().fit_transform(df_sc['Amount'].values.reshape(-1, 1))
        train, test = train_test_split(df_sc, test_size=0.3, random_state=10)
        X_train = train[train['Class'] == 0]
        X_train = X_train.drop(['Class'], axis=1)

        X_test = test.drop(['Class'], axis=1)
        y_test = test['Class']
        learning_rate = learning_rate
        input_dim = X_train.shape[1]

        input_layer = Input(shape=(input_dim,))

        encoder = Dense(16, activation='elu', activity_regularizer=regularizers.l1(learning_rate))(input_layer)
        encoder = Dense(8, activation=act)(encoder)
        encoder = Dense(4, activation=act)(encoder)

        decoder = Dense(8, activation=act)(encoder)
        decoder = Dense(16, activation=act)(decoder)
        decoder = Dense(input_dim, activation='elu')(decoder)

        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam',
                            metrics=['AUC'],
                            loss='mean_squared_error')

        EarlyStop = EarlyStopping(monitor='accuracy', patience=5, verbose=1)

        autoencoder.summary()
        history = autoencoder.fit(X_train, X_train,
                                  epochs=10,
                                  batch_size=batch_size,
                                  validation_data=(X_test, X_test),
                                  callbacks=EarlyStop,
                                  shuffle=True)
        prediction = autoencoder.predict(X_test)
        return history.history['auc']

    def svm(self, random_state=42, nu=0.1, tol=1e-4, fit_intercept=True, shuffle=True):
        print("-SVM-")
        df_sc = self.df.copy()
        df_sc['Time'] = StandardScaler().fit_transform(df_sc['Time'].values.reshape(-1, 1))
        train, test = train_test_split(df_sc, test_size=0.3, random_state=10)
        X_train = train[train['Class'] == 0]
        X_train = X_train.drop(['Class'], axis=1)
        X_test = test.drop(['Class'], axis=1)
        model_sgd = SGDOneClassSVM(random_state=random_state, nu=nu, fit_intercept=fit_intercept, shuffle=shuffle,
                                   tol=tol)
        model_sgd.fit(X_train)
        y_test = test['Class']
        pred = model_sgd.predict(X_test)
        y_pred = pred
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1
        return metrics.roc_auc_score(y_test, y_pred)

    def grid_search_svm(self):
        grid_res = []
        random_state = [42, 142, 1420, 14200, 142000, 1420000, 14200000]
        nu = [0.1, 0.01, 0.001, 0.0001, 0.0001, 0.00001]
        tol = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        fit_intercept = [True, False]
        shuffle = [True, False]
        for rand_state in random_state:
            for n in nu:
                for t in tol:
                    for interc in fit_intercept:
                        for shuf in shuffle:
                            grid_res.append({
                                'auc': self.svm(random_state=rand_state, nu=n, tol=t, shuffle=shuf,
                                                fit_intercept=interc)
                            })
        best_auc = sorted(grid_res, key=lambda d: d['auc'])[-1]
        print(f"Best auc:{best_auc['auc']}. Parameters:{best_auc}")

    def grid_search_autoencoder(self):
        grid_res = []
        learing_rates = [0.1, 0.001, 0.0001, 0.00001]
        batch_size = [64, 128, 256]
        act_func = ['sigmoid', 'tanh', 'relu', 'elu']

        for learn in learing_rates:
            for b in batch_size:
                for act in act_func:
                    print(f'params:  learning {learn}, batch-size {b}, function {act}')
                    grid_res.append(
                        {
                            'batch_size': b,
                            'learning_rates': learn,
                            'activation_function': act,
                            'auc': self.autoencoder(learn, batch_size=b, act=act)
                        }
                    )
        best_auc = sorted(grid_res, key=lambda d: d['auc'])[-1]
        print(f"Best auc:{best_auc['auc']}. Parameters:{best_auc}")

    def compute_workload_consumption(self, workload: str, cc=False, measure_power_secs=5 * 60):
        if cc:


            if os.path.exists(f"{os.getenv('CONSUMPTION_DIR')}/{workload}_consumption.csv"):
                os.remove(f"{os.getenv('CONSUMPTION_DIR')}/{workload}_consumption.csv")
            if os.path.exists(f"{os.getenv('CONSUMPTION_DIR')}/Custom_consumption.csv"):
                os.remove(f"{os.getenv('CONSUMPTION_DIR')}/Custom_consumption.csv")
            tracker = EmissionsTracker(measure_power_secs=measure_power_secs, tracking_mode='process')
            print('-Start tracking energy consumption-')
            gpu = GPU.from_utils()
            print("INFO GPU:",gpu)
            tracker.start()
            if workload == "prova":
                for i in range(1000000000):
                    a = i + i
            if workload == 'isolation_forest':
                self.grid_search_isolation_forrest()
            if workload == 'svm':
                self.grid_search_svm()
            if workload == 'autoencoder':
                self.grid_search_autoencoder()
            if workload == 'hf_sca':
                try:
                    print("HF_SCA i starting")
                    subprocess.run(['python', os.getenv('HF_SCA'), "--gpu", "0"])
                    print("HF_SCA job is completed")


                except Exception as ex:
                    print(str(ex))
            tracker.stop()
            os.rename(os.path.join(os.getenv('CONSUMPTION_DIR'), "Custom_Consumption.csv"),
                      os.path.join(os.getenv('CONSUMPTION_DIR'), f"{workload}.csv"))
        else:
            print('-Start tracking energy consumption-')

            if workload == "prova":
                for i in range(1000000):
                    a = i + i
            if workload == 'isolation_forest':
                self.grid_search_isolation_forrest()
            if workload == 'svm':
                self.grid_search_svm()
            if workload == 'autoencoder':
                self.grid_search_autoencoder()
            if workload == 'hf_sca':
                try:
                    print("HF_SCA i starting")
                    subprocess.run(['python', os.getenv('HF_SCA'), "--gpu", "0"])
                    print("HF_SCA job is completed")


                except Exception as ex:
                    print(str(ex))
