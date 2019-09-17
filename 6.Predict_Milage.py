import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


datapath= "/home/student/Student/auto-mpg.csv"
data=pd.read_csv(datapath,sep=",")

data1 = data[data.horsepower != '?']


X=data1.drop(["mpg","car name"],axis=1)
y=data1["mpg"]

x_train=X[:350]
x_test=X[350:]
y_train=y[:350]
y_test=y[350:]

linear=linear_model.LinearRegression()
linear.fit(x_train,y_train)
predicted=linear.predict(x_test)

for i,j in zip(pd.Series(predicted),y_test):
    print('Predicted=',i,'Actual=',j)

print( 'mean_sqr_eror=',mean_squared_error(predicted,y_test))

"""
OUTPUT

student@student-OptiPlex-3020:~/Documents/machine/lab/post assignment$ python lab_08.py 
('Predicted=', 33.560632551226746, 'Actual=', 33.7)
('Predicted=', 32.776348617444924, 'Actual=', 32.4)
('Predicted=', 30.87681414171141, 'Actual=', 32.9)
('Predicted=', 31.26140112825735, 'Actual=', 31.6)
('Predicted=', 26.469214016323665, 'Actual=', 28.1)
('Predicted=', 26.189095148110415, 'Actual=', 30.7)
('Predicted=', 28.741931363625856, 'Actual=', 25.4)
('Predicted=', 28.19916364070217, 'Actual=', 24.2)
('Predicted=', 23.85724781482212, 'Actual=', 22.4)
('Predicted=', 23.06360798373601, 'Actual=', 26.6)
('Predicted=', 25.94796873771953, 'Actual=', 20.2)
('Predicted=', 23.87417357166456, 'Actual=', 17.6)
('Predicted=', 29.039001616589776, 'Actual=', 28.0)
('Predicted=', 28.800780278228256, 'Actual=', 27.0)
('Predicted=', 30.28567094151568, 'Actual=', 34.0)
('Predicted=', 29.18796080337278, 'Actual=', 31.0)
('Predicted=', 29.844853060306693, 'Actual=', 29.0)
('Predicted=', 28.750162732127453, 'Actual=', 27.0)
('Predicted=', 27.722145716798206, 'Actual=', 24.0)
('Predicted=', 34.29443499114231, 'Actual=', 36.0)
('Predicted=', 35.39370950373866, 'Actual=', 37.0)
('Predicted=', 35.715869119665854, 'Actual=', 31.0)
('Predicted=', 32.147557440308404, 'Actual=', 38.0)
('Predicted=', 31.996694115016805, 'Actual=', 36.0)
('Predicted=', 34.59345045971405, 'Actual=', 36.0)
('Predicted=', 34.32991898916873, 'Actual=', 36.0)
('Predicted=', 34.236075115520705, 'Actual=', 34.0)
('Predicted=', 35.69967407089675, 'Actual=', 38.0)
('Predicted=', 35.71649789901126, 'Actual=', 32.0)
('Predicted=', 35.54492580635197, 'Actual=', 38.0)
('Predicted=', 26.745004389465887, 'Actual=', 25.0)
('Predicted=', 27.920900544156684, 'Actual=', 38.0)
('Predicted=', 29.626516427948335, 'Actual=', 26.0)
('Predicted=', 28.099877754324016, 'Actual=', 22.0)
('Predicted=', 31.71785664569541, 'Actual=', 32.0)
('Predicted=', 30.721294688978258, 'Actual=', 36.0)
('Predicted=', 27.417616929368467, 'Actual=', 27.0)
('Predicted=', 28.25606288771625, 'Actual=', 27.0)
('Predicted=', 33.82703344440547, 'Actual=', 44.0)
('Predicted=', 31.146619814601966, 'Actual=', 32.0)
('Predicted=', 29.152100767331778, 'Actual=', 28.0)
('Predicted=', 28.528093102567095, 'Actual=', 31.0)
('mean_sqr_eror=', 13.926463678743088)
"""
