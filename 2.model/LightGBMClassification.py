import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


def load_embeds(pathName, fName):
    data = np.load(pathName+fName)
    return data


def eval_f(y_pred, y_true):
    y_pred = y_pred.reshape((2, -1)).T
    y_pred = np.argmax(y_pred, axis=1)
    y_true = y_true.label
    score = metrics.f1_score(y_true, y_pred, pos_label=1)
    return 'F1_score: ', score, True


def lgb_train_model_with_split(train_x, train_y, random_seed):
    skfold = StratifiedKFold(n_splits=5, shuffle=True,
                             random_state=random_seed)
    lgb_paras = {
        'objective': 'multiclass',
        'learning_rate': 0.03,
        'num_leaves': 50,
        'lambda_l1': 0.01,
        'lambda_l2': 0.01,
        'seed': random_seed,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 4,
        'metric': 'multi_logloss',
        'num_threads': 8,
        'min_data_in_leaf': 1,
        'num_class': 2,
        # 'scale_pos_weight':100,
        'verbose': -1
    }
    # print(train_x.info())
    # print(train_y.info())
    auc, recall, precision, f1, result_proba = [], [], [], [], []
    for tr_i, val_i in skfold.split(train_x, train_y):
        tr_x, tr_y = train_x.iloc[tr_i], train_y.iloc[tr_i]
        val_x, val_y = train_x.iloc[val_i], train_y.iloc[val_i]

        train_set = lgb.Dataset(tr_x, tr_y)
        val_set = lgb.Dataset(val_x, val_y)
        lgb_model = lgb.train(lgb_paras,
                              train_set,
                              #num_boost_round=100,
                              valid_sets=[val_set],
                              early_stopping_rounds=60,
                              verbose_eval=0,
                              feval=eval_f
                              )

        val_pred = np.argmax(lgb_model.predict(
            val_x, num_iteration=lgb_model.best_iteration), axis=1)
        auc_score = metrics.roc_auc_score(val_y, val_pred)
        recall_score = metrics.recall_score(val_y, val_pred, pos_label=1)
        precision_score = metrics.precision_score(val_y, val_pred, pos_label=1)
        f1_score = metrics.f1_score(val_y, val_pred, pos_label=1)
        auc.append(auc_score)
        recall.append(recall_score)
        precision.append(precision_score)
        f1.append(f1_score)
    res = [np.mean(auc), np.mean(recall), np.mean(precision), np.mean(f1)]
    return res


def print_res(res):
    print('auc:', "{:.4f}".format(res[0]))
    print('recall:', "{:.4f}".format(res[1]))
    print('precision:', "{:.4f}".format(res[2]))
    print('F1:', "{:.4f}".format(res[3]))


def eval_f(y_pred, y_true):
    y_pred = y_pred.reshape((2, -1)).T
    y_pred = np.argmax(y_pred, axis=1)
    y_true = y_true.label
    score = metrics.f1_score(y_true, y_pred, pos_label=1)
    return 'F1_score: ', score, True

def lgb_train_model(tr_x, tr_y, val_x, val_y, random_seed):
    lgb_paras = {
        'objective': 'multiclass',
        'learning_rate': 0.03,
        'num_leaves': 10,  # 50,
        'lambda_l1': 0.01,
        'lambda_l2': 0.01,
        'seed': random_seed,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 4,
        'metric': 'multi_logloss',
        'num_threads': 8,
        'num_class': 2,
        'is_unbalance': 'True'
        #         'scale_pos_weight':100,
    }

    train_set = lgb.Dataset(tr_x, tr_y)
    val_set = lgb.Dataset(val_x, val_y)
    lgb_model = lgb.train(lgb_paras,
                          train_set,
                          num_boost_round=200,
                          valid_sets=[val_set],
                          early_stopping_rounds=60,
                          verbose_eval=-1,
                          feval=eval_f
                          )

    val_pred = np.argmax(lgb_model.predict(
        val_x, num_iteration=lgb_model.best_iteration), axis=1)

    auc_score = metrics.roc_auc_score(val_y, val_pred)
    recall_score = metrics.recall_score(val_y, val_pred, pos_label=1)
    precision_score = metrics.precision_score(val_y, val_pred, pos_label=1)
    f1_score = metrics.f1_score(val_y, val_pred, pos_label=1)

    res = [np.mean(auc_score), np.mean(recall_score),
           np.mean(precision_score), np.mean(f1_score)]

    return res


def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]
