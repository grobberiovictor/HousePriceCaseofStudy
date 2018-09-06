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


def plot_dataframe(df, columns=None, title='', y_title='', x_title='', Plot=True, campanhas=[], smooth=[False, 0.7]):
    if not columns:
        columns = df.columns.tolist()
    data = []
    for col in columns:
        if col in campanhas:
            data.append(go.Scatter(x=df.index, y=df[col], mode='lines+markers', name=col, line={'shape': 'hv'}))
        else:
            if smooth[0]:
                data.append(go.Scatter(x=df.index, y=df[col], mode='lines+markers', name=col, line={'shape': 'spline', 'smoothing': smooth[1]}))
            else:
                data.append(go.Scatter(x=df.index, y=df[col], mode='lines+markers', name=col))
    xaxis=dict(title=x_title)#, titlefont=dict(size=18))
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