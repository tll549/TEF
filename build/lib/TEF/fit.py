import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import warnings

from sklearn.model_selection import cross_validate, KFold

import sklearn.metrics

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
# from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from matplotlib.lines import Line2D


# helper functions

def data_preprocess(df, y_name, verbose, max_lev, transform_date, transform_time, impute, ml_type):
    def print_dtypes_summary(df):
        buffer = io.StringIO()
        df.info(verbose=False, buf=buffer)
        s = buffer.getvalue()
        print('   ', s.split('\n')[-3])

    # plot for y
    if verbose > 1:
        print('y:')
        if ml_type == 'classification':
            fig = plt.figure(figsize=(3, 1))
            ax = sns.countplot(df[y_name], palette=sns.color_palette("Set1"))
            totals = [i.get_height() for i in ax.patches]
            for i in ax.patches:
                ax.text(i.get_x(), i.get_height(), str(round((i.get_height()/sum(totals))*100))+'%')
        elif ml_type == 'regression':
            fig = plt.figure(figsize=(3, 1))
            ax = sns.distplot(df[y_name])
        plt.show()

    # original X information
    X = df.drop(columns=y_name)
    if verbose > 0:
        print('original X:')
        print('    shape:', X.shape)
        print_dtypes_summary(X)
        if verbose > 1:
            print('   ', X.columns.tolist())

    # clean X by dtypes
    X = X.select_dtypes(exclude=[object])
    before_dum_vars = X.columns.tolist() # this is for wanting to aggregate back all coefs after the model is trained

    to_exclude = []
    for c in X.columns:
        if X[c].dtype.name == 'category':
            if X[c].nunique() > max_lev:
                to_exclude.append(c)
        if 'datetime' in X[c].dtype.name: # notice it will get imputed as numeric next if theres NaT
            to_exclude.append(c)
            if transform_date:
                X[f'{c}_year'] = X[c].dt.year
                X[f'{c}_month'] = X[c].dt.month
                X[f'{c}_week'] = X[c].dt.week
                X[f'{c}_dayofweek'] = X[c].dt.dayofweek
            if transform_time:
                X[f'{c}_hour'] = X[c].dt.hour
                X[f'{c}_minute'] = X[c].dt.minute
                X[f'{c}_second'] = X[c].dt.second
        # exclude all null or only one lev
        if X[c].notnull().sum() == 0 or X[c].nunique() == 1:
            to_exclude.append(c)
    X = X.drop(columns=to_exclude)
    if verbose > 0:
        print('processed X:')
        print('    shape:', X.shape)
        print_dtypes_summary(X)
        if verbose > 1:
            print('   ', X.columns.tolist())

    # deal with nulls
    if impute:
        # drop null y no matter waht
        X[y_name] = df[y_name]
        X = X[X[y_name].notnull()].reset_index(drop=True)
        y = X.reset_index(drop=True)[y_name]
        X = X.drop(columns=y_name).reset_index(drop=True)
        for c in X.columns:
            if X[c].isnull().any():
                if 'float' in X[c].dtype.name or 'int' in X[c].dtype.name:
                    X[c] = X[c].fillna(X[c].median())
                else:
                    X[c] = X[c].fillna(X[c].mode().iloc[0])
    else: # dropna
        X[y_name] = df[y_name] # put y back to dropna together
        X = X.dropna()
        y = X.reset_index(drop=True)[y_name]
        X = X.drop(columns=y_name).reset_index(drop=True)

    if verbose and not impute:
        print('dropna X:')
        print('    shape:', X.shape)
        print_dtypes_summary(X)

    # plot for y again after dropna (distribution may change)
    if verbose > 0 and not impute:
        print('y:')
        if ml_type == 'classification':
            fig = plt.figure(figsize=(3, 1))
            ax = sns.countplot(df[y_name], palette=sns.color_palette("Set1"))
            totals = [i.get_height() for i in ax.patches]
            for i in ax.patches:
                ax.text(i.get_x(), i.get_height(), str(round((i.get_height()/sum(totals))*100))+'%')
        elif ml_type == 'regression':
            fig = plt.figure(figsize=(3, 1))
            ax = sns.distplot(df[y_name])
        plt.show()

    # not sure, convert y labels to 0, 1
    if y.nunique() == 2 and y.dtype != 'bool':
        y = y == y.iloc[0]

    # dummy
    X = pd.get_dummies(X)
    if verbose > 0:
        print('dummy X:')
        print('    shape:', X.shape)
        print_dtypes_summary(X)
        if verbose > 1:
            print('   ', X.columns.tolist())

    return X, y, before_dum_vars

def model_selection(X, y, verbose, models, CV, use_metric, random_state, binary, ml_type):
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    best_metric, best_model = float("-inf"), None

    for model in models:
        model_name = model.__class__.__name__
        if verbose > 1:
            print('start training', model)

        if ml_type == 'classification':
            if use_metric is None:
                use_metric = 'f1' if binary else 'f1_weighted'

            scoring = ['accuracy', use_metric]
            cv_result = cross_validate(model, X, y, scoring=scoring, 
                cv=CV, return_train_score=False, verbose=2 if verbose==2 else 0)
            accuracies = cv_result['test_accuracy']
            f1s = cv_result[f'test_{use_metric}']
            for i in range(len(accuracies)):
                entries.append((model_name, i, accuracies[i], f1s[i]))

            temp_mean_metric = np.mean(cv_result[f'test_{use_metric}'])
            if temp_mean_metric > best_metric:
                best_metric = temp_mean_metric
                best_model = model
                # best_model = models[0] # manually select the final model for debugging, logistic, a lot coefs for multiclass
                # best_model = models[1] # manually select the final model for debugging,

        elif ml_type == 'regression':
            if use_metric is None:
                use_metric = 'neg_mean_squared_error'

            scoring = ['neg_mean_squared_error', 'r2'] # ‘explained_variance’ ‘max_error’ metrics.max_error    ‘neg_mean_absolute_error’   ‘neg_mean_squared_error’    ‘neg_mean_squared_log_error’    ‘neg_median_absolute_error’ ‘r2’
            cv_result = cross_validate(model, X, y, scoring=scoring, 
                cv=CV, return_train_score=False, verbose=2 if verbose==2 else 0)
            mse = cv_result['test_neg_mean_squared_error']
            r2 = cv_result[f'test_r2']
            for i in range(len(mse)):
                entries.append((model_name, i, mse[i], r2[i]))

            temp_mean_metric = np.mean(cv_result[f'test_{use_metric}'])
            if temp_mean_metric > best_metric:
                best_metric = temp_mean_metric
                best_model = model
                # best_model = models[0] # manually select the final model for debugging

    # preparing result df
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx'] + scoring)
    if verbose > 1:
        print(cv_df)

    # plot model comparison boxplots
    if verbose > 0:
        cv_df_melted = cv_df.melt(id_vars=['model_name', 'fold_idx'], var_name='metric', value_name='score')
        g = sns.FacetGrid(cv_df_melted, col='metric', size=6, sharey=False)
        g = g.map(sns.boxplot, 'model_name', 'score', data=cv_df_melted, palette=sns.color_palette("Set2"))
        g = g.map(sns.stripplot, 'model_name', 'score', data=cv_df_melted, palette=sns.color_palette("Set2"),
            size=8, jitter=True, edgecolor="gray", linewidth=2)
        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(15)
        metric_mean = cv_df.groupby('model_name').agg('mean')

        # annotation (numbers) on plot
        if ml_type == 'classification':
            for i, l in zip(range(metric_mean.shape[0]), ax.get_xticklabels()):
                g.axes.flat[0].text(i, metric_mean.loc[l.get_text(), scoring[0]], f'{metric_mean.loc[l.get_text(), scoring[0]]*100:.2f}%', 
                    horizontalalignment='center', weight='bold', color='black')
                g.axes.flat[1].text(i, metric_mean.loc[l.get_text(), scoring[1]], f'{metric_mean.loc[l.get_text(), scoring[1]]*100:.2f}%', 
                    horizontalalignment='center', weight='bold', color='black')
        elif ml_type == 'regression':
            for i, l in zip(range(metric_mean.shape[0]), ax.get_xticklabels()):
                g.axes.flat[0].text(i, metric_mean.loc[l.get_text(), scoring[0]], f'{metric_mean.loc[l.get_text(), scoring[0]]:.0f}', 
                    horizontalalignment='center', weight='bold', color='black')
                g.axes.flat[1].text(i, metric_mean.loc[l.get_text(), scoring[1]], f'{metric_mean.loc[l.get_text(), scoring[1]]:.2f}', 
                    horizontalalignment='center', weight='bold', color='black')
        plt.show()

    return best_model

def train_test_CV_classification(X, y, verbose, best_model, CV, random_state, binary):
    if verbose > 1:
        print('random_state:', random_state)
        print('fitting best model:', best_model)
    i = 0
    y_pred_all, y_test_all = [], []
    df_coef = pd.DataFrame(index=X.columns.tolist())
    kf = KFold(n_splits=CV, shuffle=True, random_state=random_state)
    for train_index, test_index in kf.split(X):
        X_train, X_test, y_train, y_test = X.loc[train_index], X.loc[test_index], y[train_index], y[test_index]
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)

        y_pred_all += list(y_pred)
        y_test_all += list(y_test)

        if i == 0:
            confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
        else:
            confusion_matrix += pd.crosstab(y_test, y_pred)
        i += 1
            
        if not binary and 'feature_importances_' not in dir(best_model):
            for c in range(y.nunique()):
                df_coef[f'class{c}_cv{i}'] = best_model.coef_[c]
        else:
            df_coef[i] = best_model.feature_importances_ if 'feature_importances_' in dir(best_model) else best_model.coef_[0] # feature_importances_ for xgb and rf, else use coef_

    confusion_matrix /= 5
    return confusion_matrix, y_pred_all, y_test_all, df_coef

def train_test_CV_regression(X, y, verbose, best_model, CV, random_state):
    if verbose > 1:
        print('random_state:', random_state)
        print('fitting best model:', best_model)
    i = 0
    y_pred_all, y_test_all = [], []
    df_coef = pd.DataFrame(index=X.columns.tolist())
    kf = KFold(n_splits=CV, shuffle=True, random_state=random_state)
    for train_index, test_index in kf.split(X):
        X_train, X_test, y_train, y_test = X.loc[train_index], X.loc[test_index], y[train_index], y[test_index]
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)
        y_pred_all += list(y_pred)
        y_test_all += list(y_test)

        df_coef[i] = best_model.feature_importances_ if 'feature_importances_' in dir(best_model) else best_model.coef_ # feature_importances_ for xgb and rf, else use coef_
        i += 1
    return y_pred_all, y_test_all, df_coef

def classification_result(verbose, confusion_matrix, y_pred_all, y_test_all, CV, best_model_name, binary):
    if verbose > 0:
        ax = sns.heatmap(confusion_matrix, annot=True, fmt='.1f', cmap='PuBu') # color credit: Young Joo
        ax.set(title=f'mean confusion matrix of {CV}-fold {best_model_name}')
        plt.show()

    if verbose > 0:
        print('classification result:')
        print(f'    accuracy      : {sklearn.metrics.accuracy_score(y_pred_all, y_test_all)*100:.2f}')
        if binary:
            print(f'    false positive: {np.mean(np.logical_and([x==1 for x in y_pred_all], [x!=1 for x in y_test_all]))*100:>5.2f}')
            print(f'    false negative: {np.mean(np.logical_and([x!=1 for x in y_pred_all], [x==1 for x in y_test_all]))*100:>5.2f}')
            if verbose > 1:
                print(f'    precision     : {sklearn.metrics.precision_score(y_pred_all, y_test_all)*100:.2f}')
                print(f'    recall        : {sklearn.metrics.recall_score(y_pred_all, y_test_all)*100:.2f}')
            print(f'    f1            : {sklearn.metrics.f1_score(y_pred_all, y_test_all)*100:.2f}')
    if verbose > 1:
        print(sklearn.metrics.classification_report(y_test_all, y_pred_all))

def regression_result(verbose, y_pred_all, y_test_all, df_coef, best_model_name):
    y_pred_all = pd.Series(y_pred_all, name='Prediction')
    y_test_all = pd.Series(y_test_all, name='Actual')

    g = sns.jointplot(y_test_all, y_pred_all, kind='scatter', joint_kws={'alpha':0.3})
    plt.show()

    if verbose > 1:
        g = sns.jointplot(y_test_all, y_pred_all, kind='resid')
        g.fig.suptitle('Residuals')
        plt.show()

    # report
    print('regression report')
    for name, metric in {'MSE': sklearn.metrics.mean_squared_error, 
        'MAE': sklearn.metrics.mean_absolute_error,
        'R2': sklearn.metrics.r2_score,
        # 'explained_variance_score': sklearn.metrics.explained_variance_score,
        # 'median_absolute_error': sklearn.metrics.median_absolute_error,
        # 'max_error': sklearn.metrics.max_error,
        # 'mean_squared_log_error': sklearn.metrics.mean_squared_log_error,
        }.items():
        print(f'    {name:10}: {metric(y_test_all, y_pred_all):.2f}')
    # print(np.mean(np.abs(y_pred_all-y_test_all)/y_test_all))

    return

def coef_to_feat_imp(df_coef, c, use_coef, binary):
    if use_coef:
        if not binary:
            df_coef_mean = pd.DataFrame()
            for c in range(len(c)):
                df_coef_c_mean = df_coef[[cn for cn in df_coef.columns if f'class{c}' in cn]].apply('mean', axis=1)
                df_coef_mean[c] = df_coef_c_mean
            feat_imp = df_coef_mean.apply(lambda x: np.max(np.abs(x)), axis=1) # can't take into consider for sign cuz it has opposite influence for different classes
        else:
            feat_imp = df_coef.apply(lambda x: np.max(np.abs(x)), axis=1)
    else:
        feat_imp = df_coef.apply('mean', axis=1)
    return feat_imp
    
def plot_detailed_feature_importance(best_model, df_coef, binary, CV):
    '''if the model use coef, plot for every class (target)
    if the model use feature importance, plot only 1 plot'''
    use_coef = 'feature_importances_' not in dir(best_model)
    for c in range(len(best_model.classes_)):
        if not binary and use_coef:
            df_coef_c = df_coef[[cn for cn in df_coef.columns if f'class{c}' in cn]].copy()
        else:
            df_coef_c = df_coef

        df_coef_c['mean feature importance'] = df_coef_c.apply('mean', axis=1)
        df_coef_c['abs feature importance'] = np.abs(df_coef_c['mean feature importance'])
        fig = plt.figure(figsize=(10, 5))
        ax = df_coef_c.sort_values('abs feature importance', ascending=False).head(20)['abs feature importance'].plot.bar(
            color=['darkblue' if x > 0 else 'crimson' for x in df_coef_c['mean feature importance']])
        ax.legend([Line2D([0], [0], color='darkblue', lw=8), Line2D([0], [0], color='crimson', lw=8)], 
            ['positive', 'negative'])
        ax.set(title=f'mean feature importance for {best_model.classes_[c]} of {CV}-fold {best_model.__class__.__name__}', ylabel='abs feature importance')
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')

        if binary:
            break
        if not use_coef:
            ax.set(title=f'mean feature importance of {CV}-fold {best_model.__class__.__name__}')
            break

def plot_summary_feature_importance(feat_imp, best_model, CV, use_coef):
    '''deal with multi class here, sum it to one class and thus one plot only'''
    fig = plt.figure(figsize=(10, 5))
    ax = feat_imp.sort_values(ascending=False).head(20).plot.bar(color='grey')
    if use_coef:
        title = f'mean feature importance from abs max coef on all classes of {CV}-fold {best_model.__class__.__name__}'
    else:
        title = f'mean feature importance of {CV}-fold {best_model.__class__.__name__}'
    ax.set(title=title, ylabel='abs feature importance')
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    plt.show()

def plot_feature_importance(feat_imp, best_model_name, CV):
    fig = plt.figure(figsize=(10, 5))
    ax = np.abs(feat_imp).sort_values(ascending=False).head(20).plot.bar(
        color=['darkblue' if x >= 0 else 'crimson' for x in feat_imp])
    ax.legend([Line2D([0], [0], color='darkblue', lw=8), Line2D([0], [0], color='crimson', lw=8)], 
        ['positive', 'negative'])
    ax.set(title=f'mean feature importance of {CV}-fold {best_model_name}', ylabel='abs feature importance')
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')

def agg_feat_imp(feat_imp, before_dum_vars):
    agged_fi = pd.Series(index=before_dum_vars)
    for v in before_dum_vars:
        # this will have problem if the col name are similar, for example if the original column names have 'age' and 'age_man', then 'age_man' will be contained in 'age'
        agged_fi[v] = feat_imp[[v in i for i in feat_imp.index]].abs().max()
    return agged_fi


# main singal task functions

def fit_classification(df, y_name, verbose, max_lev, transform_date, transform_time, impute,
    CV, class_weight, use_metric,
    return_agg_feat_imp):
    ml_type = 'classification'

    # data preprocessing
    X, y, before_dum_vars = data_preprocess(df, y_name, verbose, max_lev, 
        transform_date, transform_time, impute, ml_type)

    # model selection
    random_state = np.random.randint(1e8)
    binary = True if y.nunique() == 2 else False
    models = [
        LogisticRegression(random_state=random_state, class_weight=class_weight, solver='lbfgs', multi_class='auto'),
        RandomForestClassifier(random_state=random_state, class_weight=class_weight, n_estimators=100),
        LinearSVC(random_state=random_state, class_weight=class_weight),
        # MultinomialNB(), # not sure if this is weighted or not
        XGBClassifier(random_state=random_state, 
            scale_pos_weight=y.value_counts(normalize=True).iloc[0] if class_weight=='balanced' and binary else 1)
    ]

    best_model = model_selection(X, y, verbose, models, CV, use_metric, random_state, binary, ml_type)
    # print(best_model)

    # final train on best model (CV)
    # best_model = [model for model in models if model.__class__.__name__ == best_model_name][0]
    random_state = np.random.randint(1e8)
    confusion_matrix, y_pred_all, y_test_all, df_coef = train_test_CV_classification(X, y, verbose, best_model, CV, random_state, binary)

    # confusion matrix and metrics
    classification_result(verbose, confusion_matrix, y_pred_all, y_test_all, CV, best_model.__class__.__name__, binary)

    # plot feature importance
    # use_coef = 'feature_importances_' not in dir(best_model)
    # plot_feature_importance(df_coef, y, binary, CV, use_coef, best_model_name)
    # plot_feature_importance(df_coef, y, binary, CV, use_coef, best_model.__class__.__name__)
    use_coef = 'feature_importances_' not in dir(best_model)
    feat_imp = coef_to_feat_imp(df_coef, best_model.classes_, use_coef, binary)
    if verbose > 0:
        plot_summary_feature_importance(feat_imp, best_model, CV, use_coef)
    if verbose > 1:
        plot_detailed_feature_importance(best_model, df_coef, binary, CV)

    if return_agg_feat_imp:
        fi = agg_feat_imp(feat_imp, before_dum_vars)
        return fi

    return


def fit_regression(df, y_name, verbose, max_lev, transform_date, transform_time, impute,
    CV, class_weight, use_metric,
    return_agg_feat_imp):
    ml_type = 'regression'

    # data preprocessing
    X, y, before_dum_vars = data_preprocess(df, y_name, verbose, max_lev, 
        transform_date, transform_time, impute, ml_type)

    # model selection
    random_state = np.random.randint(1e8)
    models = [
        LinearRegression(),
        LassoCV(random_state=random_state, cv=CV),
        RidgeCV(cv=CV), # no random_state=random_state?
        # SVR(),
        # DecisionTreeRegressor(random_state=random_state),
        RandomForestRegressor(random_state=random_state),
        XGBRegressor(random_state=random_state, objective='reg:squarederror')
    ]

    best_model = model_selection(X, y, verbose, models, CV, use_metric, random_state, None, ml_type)
    # best_model = models[0]

    # final train on best model (CV)
    random_state = np.random.randint(1e8)
    y_pred_all, y_test_all, df_coef = train_test_CV_regression(X, y, verbose, best_model, CV, random_state)

    regression_result(verbose, y_pred_all, y_test_all, df_coef, best_model.__class__.__name__)

    feat_imp = df_coef.mean(axis=1)
    plot_feature_importance(feat_imp, best_model.__class__.__name__, CV)

    if return_agg_feat_imp:
        fi = agg_feat_imp(feat_imp, before_dum_vars)
        return fi

    return


# main function

def fit(df, y_name, verbose=1, max_lev=10, transform_date=True, transform_time=False, impute=True, 
    CV=5, class_weight='balanced', use_metric=None, return_agg_feat_imp=False):

    if verbose < 2:
        warnings.simplefilter('ignore', UserWarning)
        warnings.simplefilter('ignore', FutureWarning)
        warnings.simplefilter('ignore', DeprecationWarning)
    else:
        warnings.simplefilter('always', UserWarning)
        warnings.simplefilter('always', FutureWarning)
        warnings.simplefilter('always', DeprecationWarning)

    if df[y_name].nunique() <= max_lev: # bool or category
        return fit_classification(**locals())
    elif 'float' in df[y_name].dtype.name or 'int' in df[y_name].dtype.name:
        return fit_regression(**locals())
    else:
        print('didnt handle this kind of y now')