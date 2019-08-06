__version__ = '0.0.1'

def auto_set_dtypes(df, max_num_lev=10, 
                    set_datetime=[], set_category=[], set_int=[], set_object=[], set_bool=[],
                    set_datetime_by_pattern=r'\d{4}-\d{2}-\d{2}',
                    verbose=False):
    '''
    TODO:
        figure out what is T in datetime
        let r'd{4}-d{2}-d{2}[ T]d{2}:d{2}:d{2}' becomes time part is optional
        fill ' ' with nan
        check similar strings in object or category type
        check if needs to take log
        consider those arguments and functions in descripe, summary, dtypes
    '''

    '''
    require package:
        import numpy as np
        import pandas as pd
        import re
        import io
    set to datetime if the pattern is like '2018-08-08 08:08:08', 
    it's designed for all datetime columns in a dataset have the same format like 2019-06-06 06:06:06
    set_object can be used for ID columns
    '''
    import numpy as np
    import pandas as pd
    import re
    import io

    df = df.copy() # need this or will change original df
    if verbose:
        record = pd.DataFrame({'before': df.dtypes}).transpose()

        buffer = io.StringIO()
        df.info(verbose=False, buf=buffer)
        s = buffer.getvalue()
        dtypes_before = s.split('\n')[-3]

    for c in range(df.shape[1]): # for every cols
        cur = df.iloc[:, c]
        
        if c in set_object:
            df.iloc[:, c] = cur.astype('object')
        elif c in set_datetime:
            df.iloc[:, c] = pd.to_datetime(cur, errors='coerce')
        elif c in set_bool:
            df.iloc[:, c] = cur.astype('bool')
        elif c in set_category:
            df.iloc[:, c] = cur.astype('category')
        elif c in set_int:
            df.iloc[:, c] = cur.astype('Int64') # use 'Int64' instead of int to ignore nan

        else:
            if set_datetime_by_pattern:
                if sum(cur.notnull()) > 0: # not all are null
                    fisrt_possible_date = cur[cur.notnull()].iloc[0] # use the first not null
                    if re.match(set_datetime_by_pattern, str(fisrt_possible_date)):
                        df.iloc[:, c] = pd.to_datetime(cur, errors='coerce')
                        continue
            if cur.value_counts().index.tolist() is [False, True] or \
                cur.value_counts().index.tolist() is [True, False]: # dropna=True in value_counts(), use is instead of in because in can't distinguish [True, False] and [1, 0]
                df.iloc[:, c] = cur.astype('bool')
            elif (len(cur.unique()) <= max_num_lev) and (cur.dtype.name == 'object'): # only change to category from object, in case changing for example int (2, 3, 4, 5) to category
                df.iloc[:, c] = cur.astype('category')
                
    if verbose:
        if verbose == 'summary':
            buffer = io.StringIO()
            df.info(verbose=False, buf=buffer)
            s = buffer.getvalue()
            dtypes_after = s.split('\n')[-3]
            print('before', dtypes_before)
            print('after ', dtypes_after)
        else:
            record = record.append(pd.DataFrame({'after': df.dtypes}).transpose(), sort=False)
            record = record.append(df.sample(3), sort=False)
            pd.set_option('display.max_columns', record.shape[1])
            print(record)
    return df


def dfmeta(df, max_lev=10, transpose=True, sample=True, description=None,
           style=True, color_bg_by_type=True, highlight_nan=0.5, in_cell_next_line=True,
           verbose=True, drop=None,
           check_possible_error=True, dup_lev_prop=0.7,
           save_html=None):
    '''
    version: 0.3.20190805
    documentation
        needed package:
            import numpy as np
            import pandas as pd
            import io
            from scipy.stats import skew, skewtest

        max_lev: int, the maximum acceptable number of unique levels
        transpose: bool, if True, cols is still cols
        sample: True:   sample 3 rows
                False:  don't sample
                'head': sample first 3 rows
                int:    sample first int rows
        description: dict, where keys are col names and values are description for that column
        style: bool, if True, return html, add .render() to get original html codes; 
            if False, return pandas dataframe instead and will overwrites color_bg_by_type, highlight_nan, in_cell_next_line
        color_bg_by_type: bool, coloy the cell background by dtyle, by column. will force to False if style=False
        highlight_nan: float [0, 1] or False, the proportion of when should highlight nans. will force to False if style=False
        in_cell_next_line: bool, if True, use '<br>' to separate elements in a list; if False, use ', '
        verbose: bool, whether to print the beginning shape, memory etc.
        drop: columns (or rows if transpose=True) that wants to be dropped, doesn't suppor NaNs and dtypes now
        check_possible_error: bool, check possible NaNs and duplicate levels or not
        dup_lev_prop: float [0, 1], the criteria of the repeatness of two levels
        save_html: a list with two strings elements [filename, head], e.g. ['cancelCasesDict.html', 'Cancel Cases Dictionary']
    '''
    '''
    TODO:
        have warning when the first time run
        check package imported or not
        set display.max_columns back, also set max_row (cant)
        display 0 for true 0, 0.00% for not true 0 (can't becuase need it in summary)
        possible nans should also check lower
        why sometime can calculate log skew
        detect if one col's level is contained in another col, like main_reason and detailed_reason
        check ., ' ', '  ', characters
        check strings using FuzzyWuzzy: https://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
        make it a class, have method to get dict template, get html code, save html page
    '''
    import numpy as np
    import pandas as pd
    import io
    from scipy.stats import skew, skewtest
    import re

    import warnings
    # warnings.simplefilter('ignore')
    warnings.simplefilter('ignore', RuntimeWarning) # caused from skewtest, unknown
    # warnings.simplefilter('always')

    if verbose:
        print(f'shape: {df.shape}')
        buffer = io.StringIO()
        df.info(verbose=False, buf=buffer)
        s = buffer.getvalue()
        print(s.split('\n')[-3])
        print(s.split('\n')[-2])
    
    o = pd.DataFrame(columns=df.columns)
    
    o.loc['idx'] = list(range(df.shape[1]))
    o.loc['dtype'] = df.dtypes
    
    if description is not None:
        for col, des in description.items():
            if col in df.columns.tolist():
                o.loc['description', col] = des
        o.loc['description', o.loc['description', ].isnull()] = ''
    
    if style is False:
        color_bg_by_type, highlight_nan, in_cell_next_line = False, False, False
        pd.set_option('display.max_rows', 10)
        pd.set_option('display.max_columns', 10)
    in_cell_next = "<br> " if in_cell_next_line else ", "
    
    o.loc['NaNs'] = df.apply(lambda x: f'{sum(x.isnull())}{in_cell_next}{sum(x.isnull())/df.shape[0]*100:.0f}%')

    o.loc['unique counts'] = df.apply(lambda x: f'{len(x.unique())}{in_cell_next}{len(x.unique())/df.shape[0]*100:.0f}%')
    
    def unique_index(s):
        if len(s.unique()) <= max_lev:
            o = ''
            for i in s.value_counts(dropna=False).index.tolist():
                o += str(i)
                o += '<br>' if in_cell_next_line else ', '
            return o[:-2]
        else:
            return ''
    o.loc['unique levs'] = df.apply(unique_index, result_type='expand')
    
    def print_list(l, br=', '):
        o = ''
        for e in l:
            o += e + br
        return o[:-2]
    def summary_diff_dtype(x):
        if x.dtype.name in ['object', 'bool', 'category'] and len(x.unique()) <= max_lev:
            vc = x.value_counts(dropna=False, normalize=True)
            s = ''
            for name, v in zip(vc.index, vc.values):
                s += f'{name} {v*100:>2.0f}%'
                s += '<br>' if in_cell_next_line else ', '
            return s[:-2]
        elif x.dtype.name in ['float64', 'int64']:
            o = f'quantiles: {x.quantile(q=[0, 0.25, 0.5, 0.75, 1]).values.tolist()}{in_cell_next} \
                mean: {x.mean():.2f}\
                std: {x.std():.2f} \
                cv: {x.std()/x.mean():.2f}{in_cell_next}\
                skew: {skew(x[x.notnull()]):.2f}'
            if sum(x.notnull()) > 8: # requirement of skewtest
                p = skewtest(x[x.notnull()]).pvalue
                o += f'*' if p <= 0.05 else ''
                if min(x[x!=0]) > 0 and len(x[x!=0]) > 8: # take log
                    o += f'{in_cell_next}log skew: {skew(np.log(x[x>0])):.2f}'
                    p = skewtest(np.log(x[x!=0])).pvalue
                    o += f'*' if p != p and p <= 0.05 else ''
            return o
        elif 'datetime' in x.dtype.name:
            # o = ''
            qs = x.quantile(q=[0, 0.25, 0.5, 0.75, 1]).values
            return print_list([np.datetime_as_string(q)[0:16] for q in qs], br=in_cell_next)
        else:
            return ''
    o.loc['summary'] = df.apply(summary_diff_dtype, result_type='expand') # need result_type='true' or it will all convert to object dtype
    
    if check_possible_error:
        def possible_nan(x):
            check_list = [0, ' ', 'nan', 'null']
            o = ''
            for to_check in check_list:
                if to_check == 0 :
                    if x.dtype.name != 'bool' and sum(x==0) > 0:
                        o += f' "0": {sum(x==0)}, {sum(x==0)/df.shape[0]*100:.2f}%'
                elif to_check in x.unique().tolist():
                    o += f' "{to_check}": {sum(x==to_check)}, {sum(x==to_check)/df.shape[0]*100:.2f}%'
            return o
        o.loc['possible NaNs'] = df.apply(possible_nan)

        def possible_dup_lev(x, prop=0.5):
            if x.dtype.name not in ['category', 'object']:
                return ''
            l = x.unique().tolist()
            if len(l) > 100: # maybe should adjust
                return ''
            l = [y for y in l if y == y] # remove nan
            candidate = []
            for i in range(len(l)):
                for j in range(i+1, len(l)):
                    if l[i].lower() in l[j].lower() or l[j].lower() in l[i].lower():
                        p = min(len(l[i]), len(l[j]))/max(len(l[i]), len(l[j]))
                        if p >= prop:
                            candidate.append((l[i], l[j]))
            return '; '.join(['('+', '.join(can)+')' for can in candidate])
        o.loc['possible dup lev'] = df.apply(possible_dup_lev, args=(dup_lev_prop, ))

    if sample != False:
        if sample == True and type(sample) is not int:
            sample_df = df.sample(3).sort_index()
        elif sample == 'head':
            sample_df = df.head(3)
        elif type(sample) is int:
            sample_df = df.head(sample)
        sample_df.index = ['row ' + str(x) for x in sample_df.index.tolist()]
        o = o.append(sample_df)

    if drop:
        assert 'NaNs' not in drop, 'Cannot drop NaNs for now'
        assert 'dtype' not in drop, 'Cannot drop dtype for now'
        o = o.drop(labels=drop)
        
    if transpose:
        o = o.transpose()
        o = o.rename_axis('col name').reset_index()

    if color_bg_by_type or highlight_nan != False:
        def style_rule(data, color='yellow'):
            if color_bg_by_type:
                cell_rule = 'border: 1px solid white;'
                # https://www.w3schools.com/colors/colors_picker.asp
                # saturation 92%, lightness 95%
                cmap = {'object': '#f2f2f2',
                        'datetime64[ns]': '#e7feee',
                        'int64': '#fefee7',
                        'float64': '#fef2e7',
                        'bool': '#e7fefe',
                        'category': '#e7ecfe'}
                if data.loc['dtype'].name not in cmap:
                    cell_rule += "background-color: grey"
                else:
                    cell_rule += "background-color: {}".format(cmap[data.loc['dtype'].name])
                rule = [cell_rule] * len(data)
                if transpose:
                    rule[0] = 'background-color: white;'
            else:
                rule = [''] * len(data)
            
            if float(data.loc['NaNs'][-3:-1])/100 > highlight_nan or data.loc['NaNs'][-4:] == '100%':
                rule[np.where(data.index=='NaNs')[0][0]] += '; color: red' 
            return rule
        o = o.style.apply(style_rule, axis=int(transpose)) # axis=1 for row-wise, for transpose=True
        if transpose:
            o = o.hide_index()
    
    if save_html:
        filename, head = save_html[0], save_html[1]
        r = f'<h1>{head}</h1>\n' + '<body>\n' + o.render() + '\n</body>'
        with open(filename, 'w') as f:
            f.write(r)
        return f'{filename} saved'
        # print(meta.render())
        # from IPython.display import display, HTML
        # display(HTML(meta.render()))
    return o


def plot_1var(df, max_num_lev=20, log_numeric=True):
    '''plot a plot for every cols
    TODO: for numeric, auto detect should take log or not, or just plot another log plot
    (maybe) for category, handle too many levels like object'''
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    cmap = {'object': 'grey',
            'datetime64[ns]': 'green',
            'int64': 'gold',
            'float64': 'orange',
            'bool': 'blue',
            'category': 'purple'} # https://matplotlib.org/examples/color/named_colors.html

    for c in range(df.shape[1]): # for every cols
        cur = df.iloc[:, c]
        title = f'{c}: {cur.name}, {cur.dtype.name}'
        if cur.dtype.name in ['category', 'bool']:
            if len(cur.unique()) <= max_num_lev: # skip if theres too many levels, no need when used my preprocess function
                plt.figure(figsize=(8, 2))
                color = cmap[cur.dtype.name]
                ax = cur.value_counts(dropna=False).plot(kind='bar', title=title, color=color)
                ax.set_xlabel('')
            else:
                print(f'{c}, {cur.name}, {cur.dtype.name}, has {len(cur.unique())} levels, skipped plotting')
            
            totals = [i.get_height() for i in ax.patches]
            for i in ax.patches:
                ax.text(i.get_x(), i.get_height(), str(round((i.get_height()/sum(totals))*100))+'%')
            plt.show()
        elif cur.dtype.name == 'datetime64[ns]': # should us re.search 'datetime'
            plt.figure(figsize=(8, 2))
            dsub = df[pd.notnull(cur)]
            dsub.iloc[:, c].groupby([dsub.iloc[:, c].dt.year, dsub.iloc[:, c].dt.month]) \
                .count().plot(style='.-', title=title, color=cmap[cur.dtype.name]).set_xlabel('') # orange
            plt.show()
        elif cur.dtype.name in ['int64', 'float64']:
            # fig, axes = plt.subplots(1, 2, figsize=(8, 2))
            # fig.suptitle(title)
            # ax1 = df.iloc[:, c].plot.hist(bins=20, ax=axes[0], color='C2') # green
            # ax2 = df.boxplot(df.columns[c], vert=True, ax=axes[1])
            # if log_numeric:
            #     ax1.set(yscale='log')
            #     ax2.set(yscale='log')
            # plt.show()
            fig, axes = plt.subplots(1, 2, figsize=(8, 2))
            fig.suptitle(title)
            ax1 = sns.distplot(cur[cur.notnull()], ax=axes[0], color=cmap[cur.dtype.name]) # green
            # ax2 = df.boxplot(df.columns[c], vert=True, ax=axes[1])
            ax2 = sns.boxplot(cur, color=cmap[cur.dtype.name])
            plt.show()

            if log_numeric:
                fig, axes = plt.subplots(1, 2, figsize=(8, 2))
                ax1 = sns.distplot(np.log(cur[cur > 0]), ax=axes[0], color=cmap[cur.dtype.name]) # green
                ax2 = sns.boxplot(cur, color=cmap[cur.dtype.name])
                ax1.set(xlabel=f'log {cur.name}')
                ax2.set(xscale='log', xlabel=f'log {cur.name}')
                plt.show()

            num_nans = sum(cur.isnull())/df.shape[0]*100
            nans = f'{num_nans:.2f}%' if num_nans > 1 else int(num_nans)
            print(f'quantiles: {cur.quantile(q=[0, 0.25, 0.5, 0.75, 1]).values.tolist()}, mean: {cur.mean():.2f}, NaNs: {nans}')
            if log_numeric:
                print(f'ignored {np.mean(cur==0)*100:.2f}% 0s and {nans} NaNs')
        elif cur.dtype.name == 'object':            
            plt.figure(figsize=(8, 2))
            ax = cur.value_counts(dropna=False).iloc[:20].plot(kind='bar', title=title, color=cmap[cur.dtype.name]) # cyan
            ax.set_xlabel('')
            
            totals = cur.value_counts(dropna=False).values
            for i in ax.patches:
                ax.text(i.get_x(), i.get_height(), f'{i.get_height()/sum(totals)*100:.0f}%')
            plt.show()
            print(f'{sum(totals[:20])/sum(totals)*100:.2f}% disaplyed')
        else:
            print("didn't handle this type", c, cur.dtype.name, cur.name)

def plot_1var_by_cat_y(df, y, max_num_lev=20, log_numeric=True,
    kind_for_num='boxen'):
    '''plot a plot for every cols
    TODO: 
        percentage for heatmap
        for numeric, auto detect should take log or not, or just plot another log plot
    (maybe) for category, handle too many levels like object'''
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    for c in range(df.shape[1]): # for every cols
        cur = df.iloc[:, c]
        if cur.name == y:
            continue
        title = f'{c}: {cur.name}, {cur.dtype.name}'

        if cur.dtype.name in ['category', 'bool', 'object']:
            if len(cur.unique()) <= max_num_lev: # skip if theres too many levels, no need when used my preprocess function
                ct = pd.crosstab(cur, df[y])
                ax = sns.heatmap(ct, annot = True, fmt = '.0f', cmap="YlGnBu")
                ax.set(title=title)
                plt.show()
            else:
                print(f'{c}, {cur.name}, {cur.dtype.name}, has {len(cur.unique())} levels, skipped plotting')
        elif 'datetime' in cur.dtype.name:
            print(c, 'not yet for datetime')
            # d = cur.groupby([cur.dt.year, cur.dt.month]).groups
            # print(d.head())
            # sns.lineplot()
            # print(c)
            # plt.figure(figsize=(8, 2))
            # dsub = df[pd.notnull(cur)]
            # dsub.iloc[:, c].groupby([dsub.iloc[:, c].dt.year, dsub.iloc[:, c].dt.month]) \
            #     .count().plot(style='.-', title=title, color='C1').set_xlabel('') # orange
            # plt.show()
        elif cur.dtype.name in ['int64', 'float64']:
            ax = sns.catplot(x=y, y=cur.name, data=df, kind=kind_for_num)
            ax.set(title=title)
            if log_numeric:
                ax.set(yscale='symlog')
            plt.show()
            num_nans = sum(cur.isnull())/df.shape[0]*100
            nans = f'{num_nans:.2f}%' if num_nans > 1 else int(num_nans)
            # print(f'quantiles: {cur.quantile(q=[0, 0.25, 0.5, 0.75, 1]).values.tolist()}, mean: {cur.mean():.2f}, NaNs: {nans}')
            print(f'NaNs: {nans}')
        else:
            print("didn't handle this type", c, cur.dtype.name, cur.name)