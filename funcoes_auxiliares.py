import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.preprocessing import Imputer
from scipy import stats
import warnings

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly.graph_objs import Scatter, Figure, Layout

from sklearn.metrics import mean_squared_log_error as MSLE
from sklearn.model_selection import KFold

def testa_modelo(df, model, n_splits, retorna_modelo=False, target='SalePrice'):
    
    df = df.copy()
    kf = KFold(n_splits=n_splits)
    
    list_metrics = []
    resultados = pd.DataFrame(index=df.index, columns=['y_val', 'y_prev'])
    
    for train_index, val_index in kf.split(df.index):
        X_train = df.loc[train_index,:].drop(target, axis=1).as_matrix()
        y_train = df.loc[train_index,target].tolist()
        
        X_val = df.loc[val_index].drop(target, axis=1).as_matrix()
        y_val = df.loc[val_index, target].tolist()
        
        model.fit(X_train, y_train)
        
        y_prev = model.predict(X_val)
        
        y_prev[y_prev<0] = 0
        
        resultados.loc[val_index, 'y_val'] = y_val
        resultados.loc[val_index, 'y_prev'] = y_prev
        
        list_metrics.append(np.sqrt(MSLE(y_true=y_val, y_pred=y_prev)))
        
    if retorna_modelo:
        return list_metrics, resultados, model
    else:
        return list_metrics, resultados


def plot_dataframe(df, columns=None, title='', y_title='', x_title='', Plot=True, campanhas=[], smooth=[False, 0.7]):
    if not columns:
        columns = df.columns.tolist()
    data = []
    for col in columns:
        if col in campanhas:
            data.append(go.Scattergl(x=df.index, y=df[col], mode='lines+markers', name=col, line={'shape': 'hv'}))
        else:
            if smooth[0]:
                data.append(go.Scattergl(x=df.index, y=df[col], mode='lines+markers', name=col, line={'shape': 'spline', 'smoothing': smooth[1]}))
            else:
                data.append(go.Scattergl(x=df.index, y=df[col], mode='lines+markers', name=col))
    xaxis=dict(title=x_title, tickangle=90)#, titlefont=dict(size=18))
    yaxis=dict(title=y_title)#, titlefont=dict(size=18))
    layout = go.Layout(title=title, xaxis=xaxis, yaxis=yaxis)
#     iplot({"data": data, 'layout': layout})
    if Plot:
        iplot({"data": data, 'layout': layout}, show_link=False, config={'displayModeBar': False})
    else:
        return data
    
def gbm_eval(model, X_val, y_val):
    # Plot training deviance
    # compute test set deviance
    n_estimators = model.get_params()['n_estimators']
    test_score = np.zeros((n_estimators,), dtype=np.float64)

    for i, y_pred in enumerate(model.staged_predict(X_val)):
        test_score[i] = model.loss_(y_val, y_pred)

    plt.figure(figsize=(10, 5))
    plt.title('Deviance')
    plt.plot(np.arange(n_estimators) + 1, model.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(n_estimators) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    plt.grid()

    # #############################################################################
    # Plot feature importance
    plt.figure(figsize=(20, 10))
    feature_importance = model.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, df_train.drop('SalePrice', axis=1).columns[sorted_idx].tolist())
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()