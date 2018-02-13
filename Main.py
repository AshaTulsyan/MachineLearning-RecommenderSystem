import csv
from collections import defaultdict
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVR
import scipy
# Reading from features

store_features = defaultdict(list)
cpi_sum = 0.0
cpi_count = 0.0
unemp_sum = 0.0
unemp_count = 0.0
date_map = defaultdict(int)
with open("C:/Users/krish/Desktop/CSE 258/Assignment 2/features.csv","rbU") as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    firstrow = True
    for row in reader:
        if firstrow:
            firstrow = False
            continue
        s = ''.join(row)
        l = s.split(',')
        store = l[0]
        l[2] = float(l[2])
        l[3] = float(l[3])
        if l[4]=='NA':
            l[4] = 0.0
        if l[5]=='NA':
            l[5] = 0.0
        if l[6]=='NA':
            l[6] = 0.0
        if l[7]=='NA':
            l[7] = 0.0
        if l[8]=='NA':
            l[8] = 0.0
        try:
            cpi_sum+=float(l[9])
            cpi_count+=1
        except:
            pass
        try:
            unemp_sum += float(l[10])
            unemp_count+=1
        except:
            pass
        if l[11] == 'FALSE':
            l[11] = False
        elif l[11] == 'TRUE':
            l[11] = True
        store_features[store].append(l[1:])

cpi_avg = cpi_sum/cpi_count
unemp_avg = unemp_sum/unemp_count

for s in store_features:
    count = 1
    for f in store_features[s]:
        if f[8] == 'NA':
            f[8] = cpi_avg
        else:
            f[8] = float(f[8])
        if f[9] == 'NA':
            f[9] = unemp_avg
        else:
            f[9] = float(f[9])
        date_map[f[0]] = count
        f[0] = "W" + str(count)
        count+=1

features = defaultdict(dict)
for s in store_features:
    for f in store_features[s]:
        features[s][f[0]] = f[1:]
print features['1']['W1']
# Reading from Stores
store_type = defaultdict()
store_size = defaultdict(int)
with open("C:/Users/krish/Desktop/CSE 258/Assignment 2/stores.csv","rbU") as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    firstrow = True
    for row in reader:
        if firstrow:
            firstrow = False
            continue
        s = ''.join(row)
        l = s.split(',')
        store = (l[0])
        store_type[store] = l[1]
        store_size[store] = l[2]
 # Reading from data


data = []
with open("C:/Users/krish/Desktop/CSE 258/Assignment 2/train.csv","rbU") as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    firstrow = True
    for row in reader:
        if firstrow:
            firstrow = False
            continue
        s = ''.join(row)
        l = s.split(',')
        l[3] = float(l[3])
        l[2] = "W" + str(date_map[l[2]])
        l[0] = int(l[0])
        l[1] = int(l[1])
        if l[4] == 'TRUE':
            l[4] = True
            weight = 5
        elif l[4] == 'FALSE':
            l[4] = False
            weight = 1
        l.append(weight)
        data.append(l)

from random import shuffle
shuffle(data)
size = (len(data))
training = data[:size/3]
validation = data[size/3:(2*size)/3]
testing = data[2*(size)/3:]
print len(data)
print len(training)
sum_sales_per_store = defaultdict(float)
count_sales_per_store = defaultdict(float)
for d in training:
    store = int(d[0])
    sum_sales_per_store[store]+=float(d[3])
    count_sales_per_store[store] += 1
avg_sales_per_store = defaultdict(float)
for store in sum_sales_per_store:
    avg_sales_per_store[store] = sum_sales_per_store[store]/count_sales_per_store[store]
 sum_sales_per_dept = defaultdict(float)
count_sales_per_dept = defaultdict(float)
for d in training:
    dept = int(d[1])
    sum_sales_per_dept[dept]+=float(d[3])
    count_sales_per_dept[dept] += 1
avg_sales_per_dept = defaultdict(float)
for dept in sum_sales_per_dept:
    avg_sales_per_dept[dept] = sum_sales_per_dept[dept]/count_sales_per_dept[dept]
 sum_sales_per_week = defaultdict(float)
count_sales_per_week = defaultdict(float)
for d in training:
    week = d[2]
    sum_sales_per_week[week]+=float(d[3])
    count_sales_per_week[week] += 1
avg_sales_per_week = defaultdict(float)
for week in sum_sales_per_week:
    avg_sales_per_week[week] = sum_sales_per_week[week]/count_sales_per_week[week]
# Plotting
store_dept = defaultdict(dict)
for d in data:
    store = d[0]
    dept = d[1]
    store_dept[store][dept] = d[2:]



store_sales = defaultdict(list)
for s in store_dept:
    for d in store_dept[s]:
        store_sales[s].append(store_dept[s][d][1])

        
dept_sales = defaultdict(list)
for s in store_dept:
    for d in store_dept[s]:
        dept_sales[d].append(store_dept[s][d][1])
print "Stores v/s Std. Dev. of Store Sales"
store_list = []
for s in store_sales:
    if s not in store_list:
        store_list.append(s)
sales_list = []
for s in store_list:
    sales_list.append(np.std(store_sales[s]))
plt.plot(store_list, sales_list, 'ro')
plt.show()


print "Depts v/s Std. Dev. of Dept Sales"
dept_list = []
for d in dept_sales:
    if d not in dept_list:
        dept_list.append(d)
sales_list = []
for d in dept_list:
    sales_list.append(np.std(dept_sales[d]))
plt.plot(dept_list, sales_list, 'ro')
plt.show()
print "Stores v/s Sum of Store Sales"
store_list = []
for s in store_sales:
    if s not in store_list:
        store_list.append(s)
sales_list = []
for s in store_list:
    sales_list.append(np.sum(store_sales[s]))
plt.plot(store_list, sales_list, 'ro')
plt.show()


print "Depts v/s Sum of Dept Sales"
dept_list = []
for d in dept_sales:
    if d not in dept_list:
        dept_list.append(d)
sales_list = []
for d in dept_list:
    sales_list.append(np.sum(dept_sales[d]))
plt.plot(dept_list, sales_list, 'ro')
plt.show()
# Linear Regression with Gradient Descent
def feature(d):
    feat = [1]
    store = str(d[0])
    feat.append(avg_sales_per_store[int(store)])
    dept = str(d[1])
    feat.append(avg_sales_per_dept[int(dept)])
    week = str(d[2])
    feat.append(avg_sales_per_week[week])
    for i in range(0,2):
        feat.append(float(features[store][week][i]))
    for i in range(7,9):
        feat.append(float(features[store][week][i]))
    if (d[4] == True):
        feat.append(5.0)
    elif (d[4] == False):
        feat.append(1.0)
    feat.append(int(store_size[store]))
    return feat


X = [feature(d) for d in training]
y = [float(d[3]) for d in training]
theta,residuals,rank,s = np.linalg.lstsq(X, y)
print theta

"""
Notes:-

1. First we simply tried linear regression with least squares, got a bad WMAE value of ~15000
2. Implemented gradient descent to optimize the regression, found the best value for lambda = 0.1 on the
validation set for the best WMAE value of ~11000
3. First we tried using just store and dept values as integers into the feature vector
4. Tried using each store's sales average, each dept's sales average, each week's sales average into the feature vector,
got a much better WMAE value of 8400, as given below

""";
print X[0]
# Testing on Validation Set
### Gradient descent ###
import scipy
import numpy
# Objective
def f(theta, X, y, lam):
    theta = numpy.matrix(theta).T
    X = numpy.matrix(X)
    y = numpy.matrix(y).T
    diff = X*theta - y
    diffSq = diff.T*diff
    diffSqReg = diffSq / len(X) + lam*(theta.T*theta)
    return diffSqReg.flatten().tolist()[0]

# Derivative
def fprime(theta, X, y, lam):
    theta = numpy.matrix(theta).T
    X = numpy.matrix(X)
    y = numpy.matrix(y).T
    diff = X*theta - y
    res = 2*X.T*diff / len(X) + 2*lam*theta
    return numpy.array(res.flatten().tolist()[0])

X = [feature(d) for d in validation]
y = [float(d[3]) for d in validation]
lam = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
for lamb in lam:
    theta, l, info = scipy.optimize.fmin_l_bfgs_b(f, theta, fprime, args = (X, y, lamb))
    predictions = []
    for i in range(len(validation)):
        predictions.append(np.dot(X[i], theta))
    sum = 0.0
    weight_sum = 0.0
    for i in range(len(validation)):
        sum += validation[i][4]*(abs(y[i] - predictions[i]))
        weight_sum += validation[i][4]
    print "WMAE - ", sum/weight_sum, "Lam - ", lamb
print len(predictions)
# Testing on test set
### Gradient descent ###
import scipy
import numpy
# Objective
def f(theta, X, y, lam):
    theta = numpy.matrix(theta).T
    X = numpy.matrix(X)
    y = numpy.matrix(y).T
    diff = X*theta - y
    diffSq = diff.T*diff
    diffSqReg = diffSq / len(X) + lam*(theta.T*theta)
    return diffSqReg.flatten().tolist()[0]

# Derivative
def fprime(theta, X, y, lam):
    theta = numpy.matrix(theta).T
    X = numpy.matrix(X)
    y = numpy.matrix(y).T
    diff = X*theta - y
    res = 2*X.T*diff / len(X) + 2*lam*theta
    return numpy.array(res.flatten().tolist()[0])

X = [feature(d) for d in testing]
y = [float(d[3]) for d in testing]
lam = [10]
for lamb in lam:
    theta, l, info = scipy.optimize.fmin_l_bfgs_b(f, theta, fprime, args = (X, y, lamb))
    predictions = []
    for i in range(len(testing)):
        predictions.append(np.dot(X[i], theta))
    sum = 0.0
    weight_sum = 0.0
    for i in range(len(testing)):
        sum += testing[i][4]*(abs(y[i] - predictions[i]))
        weight_sum += testing[i][4]
    print "WMAE - ", sum/weight_sum, "Lam - ", lamb

    
"""
WMAE -  9325.27779955 Lam -  1e-05
WMAE -  9325.26209252 Lam -  0.0001
WMAE -  9325.27827606 Lam -  0.001
WMAE -  9325.27388634 Lam -  0.01
WMAE -  9325.28189853 Lam -  0.1
WMAE -  9385.63392648 Lam -  1
WMAE -  9385.60441546 Lam -  10
WMAE -  9304.87674921 Lam -  100
WMAE -  9321.49523211 Lam -  1000
WMAE -  9437.64502263 Lam -  10000
""";
# SVR
X = [feature(d) for d in training]
y = [float(d[3]) for d in training]
regr = LinearSVR(C=0.0001, random_state = 0)
regr.fit(X,y)
print regr.coef_
X = [feature(d) for d in testing]
y = [float(d[3]) for d in testing]
theta = regr.coef_
predictions = []
for i in range(len(testing)):
    predictions.append(np.dot(X[i], theta))
sum = 0.0
weight_sum = 0.0
for i in range(len(testing)):
    sum += testing[i][4]*(abs(y[i] - predictions[i]))
    weight_sum += testing[i][4]
print "WMAE - ", sum/weight_sum


"""
Got the best values for C = 0.0001, WMAE of ~8100
[ -3.53683268e-01   3.23836143e-01   8.91529556e-01  -1.79479317e-01
  -6.59431761e+00  -1.33874507e+00  -2.01539405e+01  -2.58229013e+00
   7.54292063e-03   2.92419258e-02]
WMAE -  8580.1204139
""";
#Random Forest Generator
from sklearn.ensemble import RandomForestRegressor
X = [feature(d) for d in training]
y = [float(d[3]) for d in training]
regr = RandomForestRegressor(n_estimators=30).fit(X, y)

"""
Tried changing criterion value to 'mae' instead of 'mse', taking too long for training
Tried using 50 trees, memory error
Using 30 trees still gave WMAE of ~11212
""";
print regr.feature_importances_
X = [feature(d) for d in testing]
y = [float(d[3]) for d in testing]
theta = regr.feature_importances_
predictions = []
for i in range(len(testing)):
    predictions.append(np.dot(X[i], theta))
sum = 0.0
weight_sum = 0.0
for i in range(len(testing)):
    sum += X[i][8]*(abs(y[i] - predictions[i]))
    weight_sum += X[i][8]
print "WMAE - ", sum/weight_sum

"""
[ 0.          0.26448921  0.63655105  0.03061003  0.011821    0.00669866
  0.01812739  0.00966677  0.00224206  0.01979383]
WMAE -  8437.19908495
""";
# Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
X = [feature(d) for d in training]
y = [float(d[3]) for d in training]
regr = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X, y)
print regr.feature_importances_
X = [feature(d) for d in testing]
y = [float(d[3]) for d in testing]
theta = regr.feature_importances_
predictions = []
for i in range(len(testing)):
    predictions.append(np.dot(X[i], theta))
sum = 0.0
weight_sum = 0.0
for i in range(len(testing)):
    sum += X[i][8]*(abs(y[i] - predictions[i]))
    weight_sum += X[i][8]
print "WMAE - ", sum/weight_sum

"""
[ 0.    0.17  0.59  0.1   0.01  0.    0.04  0.05  0.    0.04]
WMAE -  10040.500497
""";
