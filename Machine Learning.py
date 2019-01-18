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
