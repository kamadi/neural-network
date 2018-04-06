import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

HEADERS = ["meanfreq", "sd", "median", "Q25", "Q75", "IQR", "skew", "kurt", "sp.ent", "sfm", "mode", "centroid",
           "meanfun", "minfun", "maxfun", "meandom", "mindom", "maxdom", "dfrange", "modindx", "label"]

dataset = pd.read_csv("voice.csv")

# print(dataset.head())

print(dataset.describe().transpose())

train_x, test_x, train_y, test_y = train_test_split(dataset[HEADERS[1:-1]], dataset[HEADERS[-1]])

scaler = StandardScaler()

scaler.fit(train_x)

train_x = scaler.transform(train_x)

test_x = scaler.transform(test_x)

# min = None

# for i in range(10):
clf = MLPClassifier(activation="identity",learning_rate="invscaling")
# clf = MLPClassifier(activation="logistic")
# clf = MLPClassifier(activation="tanh")
# clf = MLPClassifier(hidden_layer_sizes=(13,13),activation="relu", max_iter=300)

clf.fit(train_x, train_y)

print("Training Accuracy  :", clf.score(train_x, train_y))

print("Test Accuracy      :", clf.score(test_x, test_y))

print()
