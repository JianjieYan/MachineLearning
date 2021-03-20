from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier



url = "/home/jian/Downloads/train.csv"
names = ['1', '2', '3','4','5','6','7','8','9','10', 'Binary']
dataset = read_csv(url, names=names)

url2 = "/home/jian/Downloads/test.csv"
names2 = ['1', '2', '3','4','5','6','7','8','9','10']
dataset2 = read_csv(url2, names=names2)


#Getting the precision and recall of the model;
#Seperate trainning set to 8:2, 0.2 of training set are been used for test. just to get the precision and recall. 
#Not actual test dataset
array = dataset.values
X2 = array[:,0:10]
Y2 = array[:,10]
X2_train, X2_validation, Y2_train, Y2_validation = train_test_split(X2, Y2, test_size=0.20, random_state=1)
model =  RandomForestClassifier(n_estimators=10, max_depth=10)
model.fit(X2_train, Y2_train)# 80% of the training 
predictions2 = model.predict(X2_validation)#20% of training
print(accuracy_score(Y2_validation, predictions2))# Y2_val is the correct answer for X2_validation
print(classification_report(Y2_validation, predictions2))



#take the actual files into array and array2.
array = dataset.values
X_train = array[:,0:10]
Y_train= array[:,10]

array2=dataset2.values
X_validation= array2[:,0:10]

RFC= RandomForestClassifier(n_estimators=10, max_depth=10).fit(X_train,Y_train)
predictions = RFC.predict(X_validation)

with open("/home/jian/Downloads/labels.txt","w") as f:
  for answer in predictions:
       f.write("%d\n" %int(answer))


