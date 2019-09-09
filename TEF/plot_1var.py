import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import base64
from io import BytesIO
    
def plot_1var(df, max_lev=20, log_numeric=True, cols=None, save_plt=None):
    for c in range(df.shape[1]): # for every cols
        if cols is not None and c not in cols:
            continue # skip for col not in cols
        plot_1var_series(df, c, max_lev, log_numeric, save_plt)


def plot_1var_series(df, c, max_lev, log_numeric, save_plt,
    return_html=False):

    cmap = {'object': 'grey',
        'datetime64[ns]': 'green',
        'int8': 'gold',
        'int16': 'gold',
        'int32': 'gold',
        'int64': 'gold',
        'uint8': 'gold',
        'uint16': 'gold',
        'uint32': 'gold',
        'uint64': 'gold',
        'float16': 'orange',
        'float32': 'orange',
        'float64': 'orange',
        'bool': 'blue',
        'category': 'purple'} # https://matplotlib.org/examples/color/named_colors.html

    cur = df.iloc[:, c]
    if return_html == False:
        title = f'{c}: {cur.name}, {cur.dtype.name}'
    else:
        title = ''

    if cur.dtype.name in ['category', 'bool']:
        # plot only top 20 if theres too many levels, just like objects, should make them together in the future
        totals = cur.value_counts(dropna=False).values    
        # totals = [i.get_height() for i in ax.patches]
        if len(cur.unique()) > max_lev:
            if return_html == False:
                print(f'{sum(totals[:20])/sum(totals)*100:.2f}% disaplyed')
            else:
                title = f'{sum(totals[:20])/sum(totals)*100:.2f}% disaplyed'

        fig = plt.figure(figsize=(8, 2))
        color = cmap[cur.dtype.name]
        ax = cur.value_counts(dropna=False).iloc[:20].plot(kind='bar', title=title, color=color)
        ax.set_xlabel('')
        for i in ax.patches:
            ax.text(i.get_x(), i.get_height(), str(round((i.get_height()/sum(totals))*100))+'%')
        
        # truncate long labels
        labels = [label.get_text() if len(label.get_text()) < 20 else '...'+label.get_text()[-20:] for label in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        # rotate labels
        for label in ax.get_xticklabels():
            label.set(rotation=30, ha='right')
    elif 'datetime' in cur.dtype.name:
        fig = plt.figure(figsize=(8, 2))
        dsub = df[pd.notnull(cur)]
        dsub.iloc[:, c].groupby([dsub.iloc[:, c].dt.year, dsub.iloc[:, c].dt.month]) \
            .count().plot(style='.-', title=title, color=cmap[cur.dtype.name]).set_xlabel('') # orange
    elif 'int' in cur.dtype.name or 'float' in cur.dtype.name:
        fig, axes = plt.subplots(1, 2, figsize=(8, 2))
        fig.suptitle(title)
        ax1 = sns.distplot(cur[cur.notnull()], ax=axes[0], color=cmap[cur.dtype.name]) # green
        ax2 = sns.boxplot(cur, color=cmap[cur.dtype.name])

        if save_plt is not None:
            plt.savefig(f'{save_plt}_{c}_{cur.name}_{cur.dtype.name}_nolog.png', dpi=300, bbox_inches='tight')
        elif return_html == False:
            plt.show()

        if log_numeric:
            fig, axes = plt.subplots(1, 2, figsize=(8, 2))
            ax1 = sns.distplot(np.log(cur[cur > 0]), ax=axes[0], color=cmap[cur.dtype.name]) # green
            ax2 = sns.boxplot(cur, color=cmap[cur.dtype.name])
            ax1.set(xlabel=f'log {cur.name}')
            ax2.set(xscale='log', xlabel=f'log {cur.name}')

            if save_plt is not None:
                plt.savefig(f'{save_plt}_{c}_{cur.name}_{cur.dtype.name}_log.png', dpi=300, bbox_inches='tight')
            else:
                plt.show()

        num_nans = sum(cur.isnull())/df.shape[0]*100
        nans = f'{num_nans:.2f}%' if num_nans > 1 else int(num_nans)
        if return_html == False:
            print(f'quantiles: {cur.quantile(q=[0, 0.25, 0.5, 0.75, 1]).values.tolist()}, mean: {cur.mean():.2f}, NaNs: {nans}')
            if log_numeric:
                print(f'ignored {np.mean(cur==0)*100:.2f}% 0s and {nans} NaNs')
    elif cur.dtype.name == 'object':
        totals = cur.value_counts(dropna=False).values
        if return_html == False:
            print(f'{sum(totals[:20])/sum(totals)*100:.2f}% disaplyed')
        else:
            title = f'{sum(totals[:20])/sum(totals)*100:.2f}% disaplyed'
            
        fig = plt.figure(figsize=(8, 2))
        ax = cur.value_counts(dropna=False).iloc[:20].plot(kind='bar', title=title, color=cmap[cur.dtype.name]) # cyan
        ax.set_xlabel('')
        
        for i in ax.patches:
            ax.text(i.get_x(), i.get_height(), f'{i.get_height()/sum(totals)*100:.0f}%')

        # truncate long labels
        labels = [label.get_text() if len(label.get_text()) < 20 else '...'+label.get_text()[-20:] for label in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        # rotate labels
        for label in ax.get_xticklabels():
            label.set(rotation=30, ha='right')
    else:
        print("didn't handle this type", c, cur.dtype.name, cur.name)

    if save_plt is not None and not ('int' in cur.dtype.name or 'float' in cur.dtype.name):
        plt.savefig(f'{save_plt}_{c}_{cur.name}_{cur.dtype.name}.png', dpi=300, bbox_inches='tight')
    elif return_html and save_plt is None:
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png', transparent=True, bbox_inches='tight')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf8')
        html = '<img src=\'data:image/png;base64,{}\'></img>'.format(encoded)
        plt.close() # need to close or it will display in the end
        return html
    else:
        plt.show()