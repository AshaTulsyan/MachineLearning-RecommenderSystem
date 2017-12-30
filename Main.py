import csv
from collections import defaultdict
import numpy as np
import sklearn
import matplotlib.pyplot as plt

store_features = defaultdict(list)
cpi_sum = 0.0
cpi_count = 0.0
unemp_sum = 0.0
unemp_count = 0.0
date_map = defaultdict(int)
with open("features.csv","r") as csvfile:
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
        
for s in store_features['1']:
    print(s)
 

store_type_size = defaultdict(list)
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
        store_type_size[store] = [l[1],l[2]]
print store_type_size['1']


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

size = (len(data))
training = data[:size/2]
validation = data[size/2:(3*size)/4]
testing = data[(3*size)/4:]
print len(training), len(validation), len(testing)

print training[0]
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
print "Stores v/s Std. Dev. of Stores"
store_list = []
for s in store_sales:
    if s not in store_list:
        store_list.append(s)
sales_list = []
for s in store_list:
    sales_list.append(np.std(store_sales[s]))
plt.plot(store_list, sales_list, 'ro')
plt.show()


print "Depts v/s Std. Dev. of Depts"
dept_list = []
for d in dept_sales:
    if d not in dept_list:
        dept_list.append(d)
sales_list = []
for d in dept_list:
    sales_list.append(np.std(dept_sales[d]))
plt.plot(dept_list, sales_list, 'ro')
plt.show()
print "Stores v/s Std. Dev. of Stores"
store_list = []
for s in store_sales:
    if s not in store_list:
        store_list.append(s)
sales_list = []
for s in store_list:
    sales_list.append(np.sum(store_sales[s]))
plt.plot(store_list, sales_list, 'ro')
plt.show()


print "Depts v/s Std. Dev. of Depts"
dept_list = []
for d in dept_sales:
    if d not in dept_list:
        dept_list.append(d)
sales_list = []
for d in dept_list:
    sales_list.append(np.sum(dept_sales[d]))
plt.plot(dept_list, sales_list, 'ro')
plt.show()
print "Temperature Sales"
store_list = []
for s in store_sales:
    if s not in store_list:
        store_list.append(s)
sales_list = []
for s in store_list:
    sales_list.append(np.sum(store_sales[s]))
plt.plot(store_list, sales_list, 'ro')
plt.show()


print "Depts v/s Std. Dev. of Depts"
dept_list = []
for d in dept_sales:
    if d not in dept_list:
        dept_list.append(d)
sales_list = []
for d in dept_list:
    sales_list.append(np.sum(dept_sales[d]))
plt.plot(dept_list, sales_list, 'ro')
plt.show()

def feature(d):
    feat = [1]
    store = d[0]
    feat.append(store_features[store])
    return feat


X = [feature(d) for d in training]
y = [d[3] for d in training]
#theta,residuals,rank,s = numpy.linalg.lstsq(X, y)
