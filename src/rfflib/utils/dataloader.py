import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import os
import glob
from scipy.io import loadmat

"""
Big datasets to check:
Gas sensor array temperature modulation
WESAD (Wearable Stress and Affect Detection)
Gas sensor array under dynamic gas mixtures
PPG-DaLiA
"""
from pathlib import Path

home = str(Path.home())


def get_data(dataset_name,
             test_size=1.0 / 3.0,
             shuffle=True,
             standardize_x=True,
             standardize_y=True):
    print(f"Loading dataset {dataset_name}... ", end='')
    x, y = data_loaders[dataset_name]()

    if isinstance(x, list) and isinstance(y, list):
        # This is for datasets where the data provider has specified explicit {x,y}_{trn,tst} e.g. blog_feedback
        x_trn, x_tst, y_trn, y_tst = x + y
    else:
        x_trn, x_tst, y_trn, y_tst = train_test_split(x, y, test_size=test_size, shuffle=shuffle)

    if standardize_x:
        x_scaler = StandardScaler()
        x_scaler.fit(x_trn)
        x_trn = x_scaler.transform(x_trn)
        x_tst = x_scaler.transform(x_tst)
    else:
        x_scaler = None
    if standardize_y:
        y_scaler = StandardScaler()
        y_scaler.fit(y_trn)
        y_trn = y_scaler.transform(y_trn)
        y_tst = y_scaler.transform(y_tst)
    else:
        y_scaler = None

    return x_trn, x_tst, y_trn, y_tst, x_scaler, y_scaler


def load_elevators():
    """
    regression target:  goal
    data source:        https://web.archive.org/web/*/http://www.liacc.up.pt/~ltorgo/Regression/*
    preprocessing:      NA
    :return:
    """
    dataset_path = f"{home}/datasets/uci/elevators/elevators.data"
    data = pd.read_csv(dataset_path).values
    x = data[:, :-1]
    y = data[:, [-1]]
    return x, y


def load_kegg_directed():
    dataset_path = f"{home}/datasets/uci/KeggDirected/data.csv"
    data = pd.read_csv(dataset_path, header=None).values
    x = data[:, 2:-1]
    y = data[:, [-1]]
    return x, y


def load_kegg_undirected():
    dataset_path = f"{home}/datasets/uci/KEGGU/data.csv"
    data = pd.read_csv(dataset_path, header=None)
    data = data.replace('?', np.NaN).dropna().iloc[:, 1:].astype(np.float64)
    x = data.values[:, :-1]
    y = data.values[:, [-1]]
    return x, y


def load_house_electric():
    ## Note: N=2,000,000, D=11
    dataset_path = f"{home}/datasets/uci/HouseElectric/household_power_consumption.csv"
    data = pd.read_csv(dataset_path).values
    x = data[:, :-1]
    y = data[:, [-1]]
    return x, y


def load_song():
    """
    regression target:  The year of song release (first column)
    data source:        https://archive.ics.uci.edu/ml/datasets/Buzz+in+social+media+#
    preprocessing:      log(1+y) transform of target y
    :note:              N=500,000, D=90
    :return:
    """

    dataset_path = f"{home}/datasets/uci/Song/YearPredictionMSD.csv"
    data = pd.read_csv(dataset_path, header=None).values
    x = data[:, 1:]
    y = data[:, [0]]
    return x, y  # np.log(1+y)


def load_3d_road():
    """
    regression target:  Elevation (last column)
    data source:        https://archive.ics.uci.edu/ml/datasets/Buzz+in+social+media+#
    preprocessing:      None
    :note:              N=450,000, D=3
    :return:
    """

    dataset_path = f"{home}/datasets/uci/3DRoad/3D_spatial_network.csv"
    data = pd.read_csv(dataset_path, header=None).values
    x = data[:, :-1]
    y = data[:, [-1]]
    return x, y


def load_buzz():
    """
    regression target:  "Mean Number of active discussion (NAD). This attribute is a positive integer
                        that describe the popularity of the instance's topic. It is stored is
                        the rightmost column." from Twitter.names
    data source:        https://archive.ics.uci.edu/ml/datasets/Buzz+in+social+media+#
    preprocessing:      log(1+y) transform of target y
    :return:
    """
    dataset_path = f"{home}/datasets/uci/Buzz/Twitter.csv"
    data = pd.read_csv(dataset_path, header=None).values
    x = data[:, :-1]
    y = data[:, [-1]]
    return x, np.log(1 + y)


#
def load_blog_feedback():
    """
    regression target:  Number of comments in next 24 hours (relative to baseline)
    data source:        https://archive.ics.uci.edu/ml/datasets/BlogFeedback
    preprocessing:      NA
    notes:              This has a pre-defined train and test set (see data source)

    :return:
    """
    dataset_path = f"{home}/datasets/uci/BlogFeedback"
    data_trn = pd.read_csv(os.path.join(dataset_path, "blogData_train.csv")).values
    data_tst = []
    x_trn = data_trn[:, 0:-1]
    y_trn = data_trn[:, [-1]]

    for data_tst_path in glob.glob(dataset_path + '/blogData_test*'):
        data_tst.append(pd.read_csv(data_tst_path).values)
    data_tst = np.concatenate(data_tst, axis=0)
    x_tst = data_tst[:, 0:-1]
    y_tst = data_tst[:, [-1]]
    return [x_trn, x_tst], [y_trn, y_tst]


def load_bike_sharing_hourly():
    """
    regression target:  Number of bike shares per hour
    data source:        https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
    preprocessing:      NA
    :return:
    """
    dataset_path = f"{home}/datasets/uci/Bike-Sharing-Dataset/hour.csv"
    data = pd.read_csv(dataset_path)
    data['dteday'] = pd.to_datetime(data['dteday']).astype(np.int64) / 1000000000
    x = data.values[:, 1:-1]  # Doesn't make sense to include the sample ID
    # y = np.array(data['cnt']).reshape(-1,1)
    y = data.values[:, [-1]]
    return x, y


def load_bike_sharing_daily():
    """
    regression target:  Number of bike shares per day
    data source:        https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
    preprocessing:      NA
    :return:
    """
    dataset_path = f"{home}/datasets/uci/Bike-Sharing-Dataset/day.csv"
    data = pd.read_csv(dataset_path)
    data['dteday'] = pd.to_datetime(data['dteday']).astype(np.int64) / 1000000000
    x = data.values[:, 1:-1]  # Doesn't make sense to include the sample ID
    y = data.values[:, [-1]]
    return x, y


def load_airfoil_noise():
    """
    regression target:  sound pressure in decibels
    data source:        https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise
    preprocessing:      NA
    :return:
    """
    dataset_path = f"{home}/datasets/uci/airfoil_noise/airfoil_self_noise.dat"
    data = pd.read_csv(dataset_path, header=None, sep=r'\t+', engine='python').values
    x = data[:, 0:-1]
    y = data[:, [-1]]
    return x, y


def load_concrete_compressive():
    """
    regression target:  compressive concrete strength
    data source:        https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength
    preprocessing:      NA
    :return:
    """
    dataset_path = f"{home}/datasets/uci/concrete/Concrete_Data.xls"
    data = pd.read_excel(dataset_path).values
    x = data[:, 0:-1]
    y = data[:, [-1]]
    return x, y


def load_protein_structure():
    """
    regression target:  RMSD
    data source:        https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure
    preprocessing:      log(1+y) transform for target y
    :return:

    """
    dataset_path = f"{home}/datasets/uci/Protein/CASP.csv"
    data = pd.read_csv(dataset_path).values
    x = data[:, 1:]
    y = np.log(1 + data[:, [0]])
    return x, y


def load_superconductivity():
    """
    regression target:  Critical Temperature
    data source:        https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data
    preprocessing:      None
    :return:

    """
    dataset_path = f"{home}/datasets/uci/superconductivity/train.csv"
    data = pd.read_csv(dataset_path).values
    x = data[:, :-1]
    y = data[:, [-1]]
    return x, y


def load_ct_slice():
    """
    regression target:  Reference (relative location)
    data source:        https://archive.ics.uci.edu/ml/datasets/Relative+location+of+CT+slices+on+axial+axis
    preprocessing:      - Drop patient ID
                        - Remove columns that are constant throughout the entire available dataset
    :return:

    """
    dataset_path = f"{home}/datasets/uci/CTslice/slice_localization_data.csv"
    data = pd.read_csv(dataset_path).values
    x = data[:, 1:-1]
    y = data[:, [-1]]
    # Find columns in the entire dataset with constant values. Remove them.
    x_range = np.ptp(x, axis=0)
    const_column_idxs = np.nonzero(x_range.flatten() == 0)[0]
    x = np.delete(x, const_column_idxs, axis=1)
    return x, y


def load_parkinsons_motor():
    """
    regression target:  motor udpr
    data source:        https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring
    preprocessing:      - Drop the first 5 columns as they are not used in the original problem
    :return:

    """
    dataset_path = f"{home}/datasets/uci/parkinsons/parkinsons_updrs.csv"
    data = pd.read_csv(dataset_path).values
    x = data[:, 6:]
    y = data[:, [4]]
    return x, y


def load_parkinsons_total():
    """
    regression target:  total udpr
    data source:        https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring
    preprocessing:      - Drop the first 5 columns as they are not used in the original problem
    :return:

    """
    dataset_path = f"{home}/datasets/uci/parkinsons/parkinsons_updrs.csv"
    data = pd.read_csv(dataset_path).values
    x = data[:, 6:]
    # x = np.arcsinh(data[:, 6:])
    y = data[:, [5]]
    return x, y


def load_sarcos_out1():
    """
    regression target:  7 joint torques (last 7 dimensions)
    data source:        http://www.gaussianprocess.org/gpml/data/
    preprocessing:      NA
    notes:              Multi-output regression problem.
    :return:
    """
    dataset_path = f"{home}/datasets/sarcos"
    data_trn = loadmat(os.path.join(dataset_path, "sarcos_inv.mat"))["sarcos_inv"]
    data_tst = loadmat(os.path.join(dataset_path, "sarcos_inv_test.mat"))["sarcos_inv_test"]
    x_trn = data_trn[:, 0:-7]
    y_trn = data_trn[:, [-7]]
    x_tst = data_tst[:, 0:-7]
    y_tst = data_tst[:, [-7]]

    return [x_trn, x_tst], [y_trn, y_tst]


def load_sarcos():
    """
    regression target:  7 joint torques (last 7 dimensions)
    data source:        http://www.gaussianprocess.org/gpml/data/
    preprocessing:      NA
    notes:              Multi-output regression problem.
    :return:
    """
    dataset_path = f"{home}/datasets/sarcos"
    data_trn = loadmat(os.path.join(dataset_path, "sarcos_inv.mat"))["sarcos_inv"]
    data_tst = loadmat(os.path.join(dataset_path, "sarcos_inv_test.mat"))["sarcos_inv_test"]
    x_trn = data_trn[:, 0:-7]
    y_trn = data_trn[:, -7:]
    x_tst = data_tst[:, 0:-7]
    y_tst = data_tst[:, -7:]

    return [x_trn, x_tst], [y_trn, y_tst]


def load_abalone():
    """
    regression target:  Number of Rings
    data source:        https://web.archive.org/web/*/http://www.liacc.up.pt/~ltorgo/Regression/*
    preprocessing:      NA
    :return:
    """
    dataset_path = f"{home}/datasets/abalone/abalone.data"
    names = ['Sex', 'Length', 'Diam', 'Height', 'Whole', 'Shucked', 'Viscera', 'Shell', 'Rings']
    abalone = pd.read_csv(dataset_path, names=names)
    abalone = pd.get_dummies(abalone, drop_first=True)

    x = abalone.drop('Rings', axis=1).values.astype(dtype=np.float64)
    y = abalone['Rings'].values.astype(dtype=np.float64)
    return x, y.reshape(-1, 1)


def load_creep():
    """
    regression target:  Rupture Stress
    data source:        https://web.archive.org/web/*/http://www.liacc.up.pt/~ltorgo/Regression/*
    preprocessing:      NA
    :return:
    """
    dataset_path = f"{home}/datasets/creep/creep.data"
    names = ['Lifetime', 'Rupture_stress', 'Temperature', 'Carbon', 'Silicon', 'Manganese', \
             'Phosphorus', 'Sulphur', 'Chromium', 'Molybdenum', 'Tungsten', 'Nickel', 'Copper', \
             'Vanadium', 'Niobium', 'Nitrogen', 'Aluminium', 'Boron', 'Cobalt', 'Tantalum', 'Oxygen', \
             'Normalising_temperature', 'Normalising_time', 'Cooling_rate', 'Tempering_temperature', \
             'Tempering_time', 'Cooling_rate_tempering', 'Annealing_temperature', 'Annealing_time', \
             'Cooling_rate_annealing', 'Rhenium']
    creep = pd.read_table(dataset_path, names=names).astype('float64')

    x = creep.drop('Rupture_stress', axis=1).values.astype(dtype=np.float64)
    y = creep['Rupture_stress'].values.astype(dtype=np.float64)
    return x, y.reshape(-1, 1)


def load_ailerons():
    """
    regression target:  goal
    data source:        https://web.archive.org/web/*/http://www.liacc.up.pt/~ltorgo/Regression/*
    preprocessing:      NA
    :return:
    """
    dataset_path = f"{home}/datasets/ailerons/ailerons.data"
    names = ['climbRate', 'Sgz', 'p', 'q', 'curPitch', 'curRoll', 'absRoll', 'diffClb', 'diffRollRate', \
             'diffDiffClb', 'SeTime1', 'SeTime2', 'SeTime3', 'SeTime4', 'SeTime5', 'SeTime6', 'SeTime7', \
             'SeTime8', 'SeTime9', 'SeTime10', 'SeTime11', 'SeTime12', 'SeTime13', 'SeTime14', \
             'diffSeTime1', 'diffSeTime2', 'diffSeTime3', 'diffSeTime4', 'diffSeTime5', 'diffSeTime6', \
             'diffSeTime7', 'diffSeTime8', 'diffSeTime9', 'diffSeTime10', 'diffSeTime11', \
             'diffSeTime12', 'diffSeTime13', 'diffSeTime14', 'alpha', 'Se', 'goal']
    ailerons = pd.concat([pd.read_csv(dataset_path, names=names)]).astype('float64')

    x = ailerons.drop('goal', axis=1).values.astype(dtype=np.float64)
    y = ailerons['goal'].values.astype(dtype=np.float64)
    return x, y.reshape(-1, 1)


data_loaders = {
    # Fixed train test splits (fixed by the data source provider and/or corresponding original paper)
    "blog_feedback": load_blog_feedback,
    "sarcos": load_sarcos,
    "sarcos_out1": load_sarcos_out1,
    # Small and medium size
    "airfoil_noise": load_airfoil_noise,
    "concrete_compressive": load_concrete_compressive,
    "parkinsons_motor": load_parkinsons_motor,
    "parkinsons_total": load_parkinsons_total,
    "elevators": load_elevators,
    "bike_sharing_hourly": load_bike_sharing_hourly,
    "bike_sharing_daily": load_bike_sharing_daily,
    "protein_structure": load_protein_structure,
    "kegg_directed": load_kegg_directed,
    "ct_slice": load_ct_slice,
    "kegg_undirected": load_kegg_undirected,
    "superconductivity": load_superconductivity,

    "ailerons": load_ailerons,
    "creep": load_creep,
    "abalone": load_abalone,

    # Bigger ones
    "3d_road": load_3d_road,
    "song": load_song,
    "buzz": load_buzz,
    "house_electric": load_house_electric,
}

if __name__ == "__main__":
    """
    Example usage:
    """
    dataset_name = "blog_feedback"
    # dataset_name ="sarcos"
    # # Small and medium size
    # dataset_name ="airfoil_noise"
    # dataset_name ="concrete_compressive"
    # dataset_name ="parkinsons_motor"
    # dataset_name ="parkinsons_total"
    # dataset_name ="elevators"
    # dataset_name ="bike_sharing_hourly"
    # dataset_name ="bike_sharing_daily"
    # dataset_name ="protein_structure"
    # dataset_name ="kegg_directed"
    # dataset_name ="ct_slice"
    # dataset_name ="kegg_undirected"
    # dataset_name ="superconductivity"
    # # Bigger ones
    # dataset_name ="3d_road"
    # dataset_name ="song"
    # dataset_name ="buzz"
    # dataset_name ="house_electric"

    test_size = 1.0 / 3.0
    shuffle = True
    do_X_standardisation = True
    do_Y_standardisation = True

    X_trn, \
    X_tst, \
    Y_trn, \
    Y_tst, \
    X_scaler, \
    Y_scaler = get_data(dataset_name,
                        test_size=test_size,
                        shuffle=True,
                        standardize_x=do_X_standardisation,
                        standardize_y=do_Y_standardisation)
    N_trn, D = X_trn.shape
