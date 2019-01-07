#Laiba Shah
#12/12/2018
#Project 3: Machine Learning

'''use data in train.csv and test.csv as training and test data for the following
learning machines'''

from sklearn.model_selection import cross_val_score
from sklearn import ensemble
from sklearn import svm
from sklearn import neural_network
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,r2_score
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

mse = []
mse_clean = []
sc=[]
sc_clean = []

def MSE(actual,predicted):
    MSE = ((actual - predicted) ** 2).mean()
    return MSE

def normalize(X):
    X = StandardScaler().fit_transform(X)
    return X

def compute_measure(predicted_label,true_label):
    t_idx=(predicted_label==true_label)
    f_idx=np.logical_not(t_idx)

    p_idx=(true_label>0)
    n_idx=np.logical_not(p_idx)

    tp=np.sum(np.logical_and(t_idx,p_idx))
    tn=np.sum(np.logical_and(t_idx,n_idx))

    fp=np.sum(n_idx)-tn
    fn=np.sum(p_idx)-tp

    tp_fp_tn_fn_list=[]
    tp_fp_tn_fn_list.append(tp)
    tp_fp_tn_fn_list.append(fp)
    tp_fp_tn_fn_list.append(tn)
    tp_fp_tn_fn_list.append(fn)
    tp_fp_tn_fn_list=np.array(tp_fp_tn_fn_list)

    tp=tp_fp_tn_fn_list[0]
    fp=tp_fp_tn_fn_list[1]
    tn=tp_fp_tn_fn_list[2]
    fn=tp_fp_tn_fn_list[3]

    with np.errstate(divide='ignore'):
        sen = (1.0*tp)/(tp+fn)

    with np.errstate(divide ='ignore'):
        spc = (1.0*tn)/(tn+fp)

    with np.errstate(divide='ignore'):
        ppr = (1.0*tp)/(tp+fp)

    with np.errstate(divide='ignore'):
        npr = (1.0*tn)/(tn+fn)

    acc = (tp+tn)*1.0/(tp+fp+tn+fn)
    F1_score = 2*tp / (2*tp + fp + fn)

    ans=[]
    ans.append(acc)
    ans.append(sen)
    ans.append(spc)
    ans.append(ppr)
    ans.append(npr)
    ans.append(F1_score)

    return ans

'''Compute d-index, sensitivity, specificity, and accuracy, F-1 values for
them.'''
def classifier_metrics(method,label):
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    train = pd.DataFrame(data=train)
    train_y = train['Money_used'].values
    train = train.drop(['university','Money_used'], axis=1)
    train = train.drop(train.columns[[0]], axis=1)
    train = StandardScaler().fit_transform(train)
    train_x = pd.DataFrame(data=train)
    train_x.rename(columns={0: 'gpa', 1: 'loan', 2: 'enrollment', 3: 'edm', 4: 'grad_rate', 5: 'sat_avg', 6: 'accept_ratio',7: 'tuition', 8: 'pub_pri'}, inplace=True)


    test = pd.DataFrame(data=test)
    test_y = test['Money_used'].values
    test = test.drop(['university','Money_used'], axis=1)
    test = test.drop(test.columns[[0]], axis=1)
    test = StandardScaler().fit_transform(test)
    test_x = pd.DataFrame(data=test)
    test_x.rename(columns={0: 'gpa', 1: 'loan', 2: 'enrollment', 3: 'edm', 4: 'grad_rate', 5: 'sat_avg', 6: 'accept_ratio', 7: 'tuition', 8: 'pub_pri'}, inplace=True)

    method.fit(train_x,train_y)

    print('Predict', label,':')
    y_pred = method.predict(test_x)

    met=classification_report(test_y, y_pred)
    print("Classification report:\n", met)

    A = accuracy_score(test_y, y_pred)
    print("The accuracy is:", A)

    ans = compute_measure(y_pred, test_y)

    print('\nCheck the following classification measures: accuracy, sensitivity, specificity, PPR, NPR, F1-score\n')
    print("{}".format(ans))
    time.sleep(2)

'''Use selective learning at least for Extremely randomized trees (ET), Random
forests, and SVM (Note: this is a little bit tricky because it is a
classification problem! HINT: Convert it as a regression one)'''
def selective_learning(method, label, partition):
    train, test = train_test_split(partition, test_size=0.2, random_state=42)

    train = pd.DataFrame(data=train)
    train_y = train['Money_used'].values
    train = train.drop(['university','Money_used'], axis=1)
    train = train.drop(train.columns[[0]], axis=1)
    train = StandardScaler().fit_transform(train)
    train_x = pd.DataFrame(data=train)
    train_x.rename(columns={0: 'gpa', 1: 'loan', 2: 'enrollment', 3: 'edm', 4: 'grad_rate', 5: 'sat_avg', 6: 'accept_ratio',7: 'tuition', 8: 'pub_pri'}, inplace=True)


    test = pd.DataFrame(data=test)
    test_y = test['Money_used'].values
    test = test.drop(['university','Money_used'], axis=1)
    test = test.drop(test.columns[[0]], axis=1)
    test = StandardScaler().fit_transform(test)
    test_x = pd.DataFrame(data=test)
    test_x.rename(columns={0: 'gpa', 1: 'loan', 2: 'enrollment', 3: 'edm', 4: 'grad_rate', 5: 'sat_avg', 6: 'accept_ratio', 7: 'tuition', 8: 'pub_pri'}, inplace=True)


    method.fit(train_x,train_y)

    print('Predict', label,':')
    y_pred = method.predict(test_x)
    print(y_pred)
    test_x['Predicted']=y_pred
    test_x['Actual']=test_y

    #MSE calculation
    Mse=MSE(test_x['Actual'], test_x['Predicted'])
    print("The MSE is:\n", Mse)
    mse.append(Mse)

    score=r2_score(test_x['Actual'], test_x['Predicted'])
    print("Score is:\n", score)
    sc.append(score)

    #error calculation
    error = abs(test_x['Actual'] - test_x['Predicted'])
    test_x['Error'] = error
    y = test_x.reset_index()

    #determining the bad guys
    sorted = test_x.sort_values(by='Error', ascending=False)
    lens=int((sorted.shape[0])*(0.1))
    bad_guys= sorted.head(lens)
    #print('The Bottom 10 % aka the bad guys are: \n',bad_guys['Error'])

    #cleaning data
    sorted.drop(bad_guys.index, inplace=True)
    print(sorted.head())

    #MSE calculation for clean data
    Mse_clean = MSE(sorted['Actual'], sorted['Predicted'])
    print("The MSE after cleaning is:\n", Mse_clean)
    mse_clean.append(Mse_clean)

    score_clean=r2_score(sorted['Actual'], sorted['Predicted'])
    print("Score is:\n", score_clean)
    sc_clean.append(score_clean)

def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    #classifiers
    etc = ensemble.ExtraTreesClassifier(n_estimators=10,max_features= 9,criterion= 'entropy',min_samples_split=5,max_depth= 50, min_samples_leaf= 5)
    rfc = ensemble.RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    svc = svm.SVC(kernel='rbf', tol=0.0001, gamma=0.5, C=1)
    gbc = ensemble.GradientBoostingClassifier(n_estimators=5000, max_depth=8, min_samples_split=2, learning_rate=0.005)
    mlpc = neural_network.MLPClassifier(hidden_layer_sizes=(200, 200))

    #regressors
    etr = ensemble.ExtraTreesRegressor(n_estimators=10,max_features= 9, min_samples_split=5,max_depth= 50, min_samples_leaf= 5)
    rfr = ensemble.RandomForestRegressor(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    svr = svm.SVR(kernel='rbf', tol=0.0001, gamma=0.5, C=1)
    gbr = ensemble.GradientBoostingRegressor(n_estimators=5000, max_depth=8, min_samples_split=2, learning_rate=0.005)
    #mlpr = neural_network.MLPRegressor(hidden_layer_sizes=(200, 200))

    class_method=[etc,rfc,svc,gbc,mlpc]
    reg_method=[etr,rfr,svr,gbr]
    label=['ERT','RF','SVM','GB','MLP']

    for i in range(len(class_method)):
        classifier_metrics(class_method[i], label[i])
    for s in range(len(reg_method)):
        selective_learning(reg_method[s], label[s], train)
        selective_learning(reg_method[s], label[s], test)

    MSE = pd.DataFrame({'MSE_Raw': mse, 'MSE_Clean': mse_clean})
    MSE.rename(index={0: 'ERT Train', 1: 'ERT Test',2: 'RF Train', 3: 'RF Test', 4:'SVM Train',5:'SVM Test', 6:'GB Train',7:'GB Test', 8:'MLP Train',9:'MLP Test'}, inplace=True)
    print(MSE)
    MSE.plot.bar(rot=0)
    plt.title("MSE Training Data and Test Data")
    plt.legend()
    plt.show()

    SCORE = pd.DataFrame({'R2 Score_Raw': sc, 'R2 Score_Clean': sc_clean})
    SCORE.rename(index={0: 'ERT Train', 1: 'ERT Test',2: 'RF Train', 3: 'RF Test', 4:'SVM Train',5:'SVM Test', 6:'GB Train',7:'GB Test', 8:'MLP Train',9:'MLP Test'}, inplace=True)
    print(SCORE)
    SCORE.plot.bar(rot=0)
    plt.title("R2 Score Training Data and Test Data")
    plt.legend()
    plt.show()

main()


'''Output:

Predict ERT :
Classification report:
               precision    recall  f1-score   support

           0       0.71      0.36      0.48        14
           1       0.00      0.00      0.00        12
           2       0.68      0.98      0.80        44

   micro avg       0.69      0.69      0.69        70
   macro avg       0.47      0.44      0.43        70
weighted avg       0.57      0.69      0.60        70

The accuracy is: 0.6857142857142857

Check the following classification measures: accuracy, sensitivity, specificity, PPR, NPR, F1-score

[0.6857142857142857, 0.7678571428571429, 0.35714285714285715, 0.8269230769230769, 0.2777777777777778, 0.7962962962962963]

Predict RF :
Classification report:
               precision    recall  f1-score   support

           0       0.43      0.43      0.43        14
           1       0.20      0.08      0.12        12
           2       0.71      0.82      0.76        44

   micro avg       0.61      0.61      0.61        70
   macro avg       0.44      0.44      0.43        70
weighted avg       0.56      0.61      0.58        70

The accuracy is: 0.6142857142857143

Check the following classification measures: accuracy, sensitivity, specificity, PPR, NPR, F1-score

[0.6142857142857143, 0.6607142857142857, 0.42857142857142855, 0.8222222222222222, 0.24, 0.7326732673267327]

Classification report:
               precision    recall  f1-score   support

           0       0.29      0.14      0.19        14
           1       0.00      0.00      0.00        12
           2       0.63      0.91      0.75        44

   micro avg       0.60      0.60      0.60        70
   macro avg       0.31      0.35      0.31        70
weighted avg       0.46      0.60      0.51        70

The accuracy is: 0.6

Check the following classification measures: accuracy, sensitivity, specificity, PPR, NPR, F1-score

[0.6, 0.7142857142857143, 0.14285714285714285, 0.7692307692307693, 0.1111111111111111, 0.7407407407407407]

Predict GB :
Classification report:
               precision    recall  f1-score   support

           0       0.50      0.29      0.36        14
           1       0.62      0.42      0.50        12
           2       0.72      0.89      0.80        44

   micro avg       0.69      0.69      0.69        70
   macro avg       0.62      0.53      0.55        70
weighted avg       0.66      0.69      0.66        70

The accuracy is: 0.6857142857142857

Check the following classification measures: accuracy, sensitivity, specificity, PPR, NPR, F1-score

[0.6857142857142857, 0.7857142857142857, 0.2857142857142857, 0.8148148148148148, 0.25, 0.8]

Predict MLP :
Classification report:
               precision    recall  f1-score   support

           0       0.50      0.50      0.50        14
           1       0.12      0.08      0.10        12
           2       0.69      0.75      0.72        44

   micro avg       0.59      0.59      0.59        70
   macro avg       0.44      0.44      0.44        70
weighted avg       0.55      0.59      0.57        70

The accuracy is: 0.5857142857142857

Check the following classification measures: accuracy, sensitivity, specificity, PPR, NPR, F1-score

[0.5857142857142857, 0.6071428571428571, 0.5, 0.8292682926829268, 0.2413793103448276, 0.7010309278350515]

Predict ERT :
[1.90138889 1.06289683 1.45547619 1.39234127 1.80625902 1.93253968
 1.33126782 1.69541667 1.5156746  0.86881666 1.13464286 1.10707718
 1.29373016 0.45760454 1.57013889 1.97777778 1.76095238 1.92003968
 1.60150794 1.56380952 1.17183908 1.32928571 0.62712835 1.64779221
 0.59780062 1.48444444 1.28342638 1.52424603 1.15125    1.10087535
 1.44230159 1.77411616 1.24399184]
The MSE is:
 0.4707910593157379
Score is:
 0.32362603747382757
         gpa      loan  enrollment    ...     Predicted  Actual     Error
28 -0.683400  0.798959   -0.740440    ...      1.151250       0  1.151250
10  1.038142 -0.005048    0.914912    ...      1.134643       0  1.134643
29 -0.253015  0.456341    0.651895    ...      1.100875       0  1.100875
11 -1.544172 -3.446702    0.281459    ...      1.107077       2  0.892923
20 -1.544172 -1.665282    0.739019    ...      1.171839       2  0.828161

[5 rows x 12 columns]
The MSE after cleaning is:
 0.31616571123934933
Score is:
 0.4538404220433504

Predict ERT :
[1.22585498 1.43571429 1.833171   1.88809524 1.56359307 1.02809524
 1.10857143 1.82464286 0.9097619  1.86650433 1.75       0.86611111
 1.71130952 1.31666667]
The MSE is:
 0.7360225013305969
Score is:
 -0.16339040532900784
         gpa      loan  enrollment    ...     Predicted  Actual     Error
13 -1.222983  0.770816   -0.798396    ...      1.316667       0  1.316667
0  -0.324203  0.589760   -0.331019    ...      1.225855       0  1.225855
11  1.023968 -0.082947    1.682661    ...      0.866111       2  1.133889
8  -0.324203 -0.345739    1.372306    ...      0.909762       2  1.090238
10 -0.414081 -0.437018    0.044999    ...      1.750000       1  0.750000

[5 rows x 12 columns]
The MSE after cleaning is:
 0.5246520468028043
Score is:
 0.036237000981805045

Predict RF :
[1.6 1.5 1.3 1.8 1.4 1.8 1.7 1.8 1.4 1.1 1.5 0.9 0.6 0.2 1.7 2.  1.6 1.8
 2.  1.2 1.6 0.8 0.1 1.9 0.3 1.2 1.1 1.3 1.7 1.  0.9 1.4 0.7]
The MSE is:
 0.7584848484848484
Score is:
 -0.08969656992084452
         gpa      loan  enrollment  ...    Predicted  Actual  Error
10  1.038142 -0.005048    0.914912  ...          1.5       0    1.5
12  0.607757  0.975167   -0.703872  ...          0.6       2    1.4
21  0.650795  0.731278   -0.902894  ...          0.8       2    1.2
26 -1.544172 -0.044191    0.065432  ...          1.1       0    1.1
30 -0.683400  0.685193   -1.001720  ...          0.9       2    1.1

[5 rows x 12 columns]
The MSE after cleaning is:
 0.5336666666666667
Score is:
 0.0781190019193857

Predict RF :
[0.8 0.7 1.8 1.4 1.7 0.1 0.5 1.9 0.8 1.7 1.8 0.9 0.8 0.8]
The MSE is:
 0.7535714285714287
Score is:
 -0.19112903225806455
         gpa      loan  enrollment  ...    Predicted  Actual  Error
8  -0.324203 -0.345739    1.372306  ...          0.8       2    1.2
12 -0.773593  0.270475   -0.254055  ...          0.8       2    1.2
11  1.023968 -0.082947    1.682661  ...          0.9       2    1.1
5  -1.222983  0.634844   -0.815169  ...          0.1       1    0.9
0  -0.324203  0.589760   -0.331019  ...          0.8       0    0.8

[5 rows x 12 columns]
The MSE after cleaning is:
 0.5892307692307692
Score is:
 -0.08239130434782616

Predict SVM :
[1.8782645  0.83772862 1.76093406 1.78568699 1.89052146 1.76470721
 1.42307034 1.90925957 1.89766745 0.69514881 1.47449663 0.97045326
 1.51716967 0.21119527 1.64483064 1.84132961 2.01676803 1.99330655
 1.90178441 1.54934235 1.38335034 1.59418723 0.59950782 1.69217881
 0.71478547 1.76664965 1.64097744 1.98681304 0.79102625 1.22632229
 1.80846874 1.71950787 0.99673811]
The MSE is:
 0.5175628233252643
Score is:
 0.2564301918189804
         gpa      loan  enrollment    ...     Predicted  Actual     Error
10  1.038142 -0.005048    0.914912    ...      1.474497       0  1.474497
29 -0.253015  0.456341    0.651895    ...      1.226322       0  1.226322
11 -1.544172 -3.446702    0.281459    ...      0.970453       2  1.029547
28 -0.683400  0.798959   -0.740440    ...      0.791026       0  0.791026
2  -0.683400 -0.005859   -0.214223    ...      1.760934       1  0.760934

[5 rows x 12 columns]
The MSE after cleaning is:
 0.2830873445609049
Score is:
 0.5109815545013161

Predict SVM :
[1.51885204 1.99319925 1.9557138  1.67284312 1.66564355 1.47516176
 1.87163888 1.68107169 0.71524775 1.86723971 1.73842908 1.35904007
 1.57211676 1.47333598]
The MSE is:
 0.9320338756819765
Score is:
 -0.47321483575538204
         gpa      loan  enrollment    ...     Predicted  Actual     Error
0  -0.324203  0.589760   -0.331019    ...      1.518852       0  1.518852
13 -1.222983  0.770816   -0.798396    ...      1.473336       0  1.473336
8  -0.324203 -0.345739    1.372306    ...      0.715248       2  1.284752
1  -0.324203  0.755657   -0.740227    ...      1.993199       1  0.993199
6  -1.672373  1.058642   -0.429515    ...      1.871639       1  0.871639

[5 rows x 12 columns]
The MSE after cleaning is:
 0.7355300102541028
Score is:
 -0.35113664927112365

Predict GB :
[ 1.96486075e+00  1.82836708e+00  1.91758750e+00  1.87184578e+00
  1.84008067e+00  2.00020634e+00  1.57808104e+00  2.01683823e+00
  1.92577011e+00  9.90269598e-01  1.99837241e+00  1.52558886e-01
  5.78487499e-02 -3.86391560e-05  1.99784016e+00  2.00022152e+00
  1.90567847e+00  2.00023327e+00  1.92921497e+00  1.18195729e+00
  1.76241913e+00  1.26373180e+00  7.06970631e-03  2.00034948e+00
  2.09854596e-02  4.38865309e-02  1.70133074e+00  1.59943101e+00
  1.44345707e+00  7.19892748e-01  1.72304024e+00  7.76527681e-01
  3.35585726e-01]
The MSE is:
 1.069932013800369
Score is:
 -0.5371450699585778
         gpa      loan  enrollment    ...     Predicted  Actual     Error
12  0.607757  0.975167   -0.703872    ...      0.057849       2  1.942151
3  -1.544172  0.644077   -0.808274    ...      1.871846       0  1.871846
11 -1.544172 -3.446702    0.281459    ...      0.152559       2  1.847441
26 -1.544172 -0.044191    0.065432    ...      1.701331       0  1.701331
28 -0.683400  0.798959   -0.740440    ...      1.443457       0  1.443457

[5 rows x 12 columns]
The MSE after cleaning is:
 0.7832172984550122
Score is:
 -0.23883228226627606

Predict GB :
[1.44968356e+00 5.70989504e-01 1.33743036e+00 1.00066654e+00
 1.97503270e+00 5.71025788e-01 1.44965105e+00 1.99978823e+00
 4.56353039e-04 1.99980399e+00 1.99976753e+00 4.60559447e-04
 1.18478215e+00 5.70952908e-01]
The MSE is:
 1.2925555657534848
Score is:
 -1.043071700707121
         gpa      loan  enrollment    ...     Predicted  Actual     Error
8  -0.324203 -0.345739    1.372306    ...      0.000456       2  1.999544
11  1.023968 -0.082947    1.682661    ...      0.000461       2  1.999539
0  -0.324203  0.589760   -0.331019    ...      1.449684       0  1.449684
10 -0.414081 -0.437018    0.044999    ...      1.999768       1  0.999768
3   0.889151 -1.719566   -0.115710    ...      1.000667       2  0.999333

[5 rows x 12 columns]
The MSE after cleaning is:
 1.084350918235747
Score is:
 -0.99190549110697
            MSE_Raw  MSE_Clean
ERT Train  0.470791   0.316166
ERT Test   0.736023   0.524652
RF Train   0.758485   0.533667
RF Test    0.753571   0.589231
SVM Train  0.517563   0.283087
SVM Test   0.932034   0.735530
GB Train   1.069932   0.783217
GB Test    1.292556   1.084351
           R2 Score_Raw  R2 Score_Clean
ERT Train      0.323626        0.453840
ERT Test      -0.163390        0.036237
RF Train      -0.089697        0.078119
RF Test       -0.191129       -0.082391
SVM Train      0.256430        0.510982
SVM Test      -0.473215       -0.351137
GB Train      -0.537145       -0.238832
GB Test       -1.043072       -0.991905

Process finished with exit code 0'''
