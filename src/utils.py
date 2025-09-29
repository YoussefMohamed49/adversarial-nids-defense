# src/utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def load_and_preprocess_nsl_kdd(data_path='../data/'):
    """Loads and preprocesses the NSL-KDD dataset."""
    print("Loading and preprocessing NSL-KDD dataset...")
    # ... (The full data loading code is identical to the previous versions) ...
    # (Identical function from previous steps)
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]
    df_train = pd.read_csv(data_path + 'KDDTrain+.txt', header=None, names=columns)
    df_test = pd.read_csv(data_path + 'KDDTest+.txt', header=None, names=columns)
    df_train.drop('difficulty', axis=1, inplace=True); df_test.drop('difficulty', axis=1, inplace=True)
    label_mapping = {'normal':'normal', 'back':'dos', 'land':'dos', 'neptune':'dos', 'pod':'dos', 'smurf':'dos', 'teardrop':'dos','mailbomb':'dos', 'apache2':'dos', 'processtable':'dos', 'udpstorm':'dos', 'ipsweep':'probe', 'nmap':'probe','portsweep':'probe', 'satan':'probe', 'mscan':'probe', 'saint':'probe', 'ftp_write':'r2l', 'guess_passwd':'r2l','imap':'r2l', 'multihop':'r2l', 'phf':'r2l', 'spy':'r2l', 'warezclient':'r2l', 'warezmaster':'r2l','sendmail':'r2l', 'named':'r2l', 'snmpgetattack':'r2l', 'snmpguess':'r2l', 'xlock':'r2l', 'xsnoop':'r2l','worm':'r2l', 'buffer_overflow':'u2r', 'loadmodule':'u2r', 'perl':'u2r', 'rootkit':'u2r','httptunnel':'u2r', 'ps':'u2r', 'sqlattack':'u2r', 'xterm':'u2r'}
    df_train['label'] = df_train['label'].map(label_mapping); df_test['label'] = df_test['label'].map(label_mapping)
    df_train.dropna(inplace=True); df_test.dropna(inplace=True)
    categorical_cols = ['protocol_type', 'service', 'flag']
    df_full = pd.concat([df_train, df_test], axis=0)
    df_full_encoded = pd.get_dummies(df_full, columns=categorical_cols, dummy_na=False)
    df_train_encoded = df_full_encoded.iloc[:len(df_train)]; df_test_encoded = df_full_encoded.iloc[len(df_train):]
    X_train = df_train_encoded.drop('label', axis=1); y_train_str = df_train_encoded['label']
    X_test = df_test_encoded.drop('label', axis=1); y_test_str = df_test_encoded['label']
    scaler = MinMaxScaler(); X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)
    label_encoder = LabelEncoder(); y_train = label_encoder.fit_transform(y_train_str); y_test = label_encoder.transform(y_test_str)
    return (X_train.astype(np.float32), y_train), (X_test.astype(np.float32), y_test)

def create_baseline_model(input_shape, num_classes):
    """Creates and compiles the Keras MLP model."""
    model = Sequential([
        Input(shape=input_shape),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model