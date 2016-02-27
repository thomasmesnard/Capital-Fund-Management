import numpy
import matplotlib.pyplot as plt
import csv
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime


proportion_train = 0.95
file_name = 'training_input.csv'
target_file = 'challenge_output_data_training_file_prediction_of_transaction_volumes_in_financial_markets.csv'

data = numpy.genfromtxt(file_name,skip_header=1,delimiter=',',filling_values=1)
product = data[:,2]
data = data[:,3:].astype(numpy.float32)

ref = {int(k):i for i, k in enumerate(open('listid.txt'))}
product = numpy.array([ref[product[i]] for i in range(len(product))])
product = numpy.equal(product[:,None],numpy.arange(len(ref))[None,:])

target = numpy.genfromtxt(target_file,skip_header=1,delimiter=';',filling_values=1)
target = target[:,1].astype(numpy.float32)
target = target + numpy.equal(target,0)

n_train  = int(proportion_train*data.shape[0])

train_x = numpy.concatenate((data[:n_train,:],product[:n_train,:]),axis=1)
train_y = target[:n_train]

valid_x = numpy.concatenate((data[n_train:,:],product[n_train:,:]),axis=1)
valid_y = target[n_train:]

print "Random Forest:"
rf = RandomForestRegressor(n_estimators=200,n_jobs=4)
rf.fit(train_x, train_y)

print "Doing validation:"
valid_yhat = rf.predict(valid_x)
print "Error rate: "
print numpy.mean(abs((valid_yhat-valid_y)/valid_y))

print "Predicting for test:"
test = numpy.genfromtxt('testing_input.csv',skip_header=1,delimiter=',',filling_values=1)
test_input = test[:,3:].astype(numpy.float32)

test_product = test[:,2]
test_product = numpy.array([ref[test_product[i]] for i in range(len(test_product))])
test_product = numpy.equal(test_product[:,None],numpy.arange(len(ref))[None,:])

test_id = test[:,0]

test_x = numpy.concatenate((test_input,test_product),axis=1)

test_yhat = rf.predict(test_x)

with open('results.csv', 'wb') as csvfile:
    print "Writing results on test set..."
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(['ID','TARGET'])
    for i in range(len(test_yhat)):
        csvwriter.writerow([int(test_id[i]), test_yhat[i]])
