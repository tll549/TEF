import re
import pandas as pd
import matplotlib.pyplot as plt

from .auto_set_dtypes import auto_set_dtypes
# from auto_set_dtypes import auto_set_dtypes # local only

def load_dataset(name, **kws):
    '''
    maybe cache in the future https://github.com/mwaskom/seaborn/blob/master/seaborn/utils.py
    '''
    if name == 'titanic_raw':
        filename = 'titanic'
    else:
        filename = name

    full_path = f'https://raw.githubusercontent.com/tll549/TEF/master/data/{filename}.csv'
    df = pd.read_csv(full_path, **kws)

    if name == 'titanic':
        df = auto_set_dtypes(df, set_object=['passenger_id'], verbose=0)    
    return df
    

def reorder_col(df, to_move, after=None, before=None):
    assert after is not None or before is not None, 'need sth'
    cols = df.columns.tolist()
    assert to_move in cols, f'{to_move} not in column names'
    
    cols = [x for x in cols if x != to_move] # remove to_move
    # insert back
    if after:
        assert after in cols, f'{after} not in column names'
        cols.insert(cols.index(after)+1, to_move)
    elif before:
        assert before in cols, f'{before} not in column names'
        cols.insert(cols.index(before), to_move)
    return df[cols]

def rename_cols_by_words(df, words=[], mapper={}, verbose=1):
    '''replace white space as _, make sure words are separated by _, lower case all'''
    df2 = df.copy()
    no_change = []
    
    if len(mapper) > 0:
        df2 = df2.rename(columns=mapper)
    for c in range(df2.shape[1]):
        cn_original = df2.columns[c]
        cn = cn_original.lower()
        if cn not in list(mapper.keys()):
            if ' ' in cn:
                df2.rename(columns={cn_original: cn.replace(' ', '_')}, inplace=True)
            for w in words:
                if w in cn: # if abwcd, wabcd, abcdw, becomes ab_w_cd, w_abcd, abcd_w
                    if re.search('^'+w, cn):
                        if not re.search('^'+w+'_', cn): # wabcd, not w_abcd
                            cn = cn.replace(w, w+'_')
                    elif re.search(w+'$', cn):
                        if not re.search('_'+w+'$', cn): # abcdw, not abcd_w
                            cn = cn.replace(w, '_'+w)
                    else: # abwcd or ab_wcd or abw_cd
                        if re.search('_'+w, cn) and not re.search(w+'_', cn): # ab_wcd
                            cn = cn.replace(w, w+'_')
                        elif re.search(w+'_', cn) and not re.search('_'+w, cn): # abw_cd
                            cn = cn.replace(w, '_'+w)
                        elif not re.search(w+'_', cn) and not re.search('_'+w, cn): # abwcd
                            cn = cn.replace(w, '_'+w+'_')

            df2.rename(columns={cn_original: cn}, inplace=True)
        if verbose > 0:
            if df.columns[c] != df2.columns[c]:
                print(f'{c:<3}, {df.columns[c]:25} -> {df2.columns[c]:25}')
            else:
                no_change.append(c)
    if verbose > 1:
        if len(no_change) > 0:
            print("didn't changed:", [df2.columns[c] for c in no_change])
    return df2


def ct(s1, s2, style=True, col_name=None, sort=False, head=False):
    '''
    crosstab count and percentage
    sort should be using the same name as col_name
    it is always 
        row sums to 1, which is s1
        total counts only on row
        color background by columns
    '''
    # to avoid s1 or s2 is a condition
    if s1.name is None:
        s1.name = 's1'
    if s2.name is None:
        s2.name = 's2'

    c1 = pd.crosstab(s1, s2, margins=True)
    c2 = pd.crosstab(s1, s2, normalize='index')*100

    if col_name is not None:
        c1.columns = col_name + ['All']
        c2.columns = col_name
        
    o = pd.concat([c1, c2], axis=1, keys=['count', 'proportion'], sort=False)
    o.index.name = s1.name
    o = o[o.index != 'All'] # remove the sum from margins for row, in order to style and sort
    o.columns.names = [None, None]

    # add a highest column name for s2
    o = pd.concat([o], keys=[s2.name], names=[None], axis=1)

    if sort:
        if sort == True:
            sort = (s2.name, 'count', 'All')
        o = o.sort_values(sort, ascending=False)
    if head:
        o = o.head(head)
    if style:
        o = o.style.format('{:.0f}').background_gradient(axis=0)
    return o


def set_relation(s1, s2, plot=True):    
    sr = pd.Series()
    sr['s1 orig len'] = len(s1)
    sr['s2 orig len'] = len(s2)
    s1 = s1[s1.notnull()]
    s2 = s2[s2.notnull()]
    sr['s1 notnull len'] = len(s1)
    sr['s2 notnull len'] = len(s2)
    sr['s1 nunique'] = s1.nunique()
    sr['s2 nunique'] = s2.nunique()
    sr['union'] = len(set(s1) | set(s2))
    sr['intersection'] = len(set(s1) & set(s2))
    sr['in s1 only'] = len(set(s1) - set(s2))
    sr['in s2 only'] = len(set(s2) - set(s1))
    
    if plot:
        sr_color = []
        for n in sr.index:
            if 's1' in n:
                sr_color.append('darkblue')
            elif 's2' in n:
                sr_color.append('crimson')
            else:
                sr_color.append('purple')
        ax = sr.plot.bar(color=sr_color)
        for label in ax.get_xticklabels():
            label.set_rotation(20)
            label.set_ha('right')
        totals = sr.value_counts(dropna=False).values
        for i in ax.patches:
            ax.text(i.get_x(), i.get_height(), f'{i.get_height()}')
        ax.set(title=f'set relation between {s1.name} & {s2.name}')
        plt.show()
    
    return sr


def correspondence(s1, s2, verbose=1, fillna=True):
    '''
    credit: Chandra Kuma
    [1,2,3,4,5]
    [1,2,3,4,5]
    '1:1': 5
    
    [1,2,3,4,5]
    [2,3,4,5,6]
    'None': 5
    
    [1,2,3,4,5]
    [6,6,6,6,6]
    'm:1': 5
    
    [6,6,6,6,6]
    [1,2,3,4,5]
    '1:m': 5
    '''
    
    # imput nan, because nan != nan
    if fillna and isinstance(s1, pd.core.series.Series):
        s1 = s1.fillna('nan_filled')
        s2 = s2.fillna('nan_filled')
    
    def scan(s1, s2):
        d = {}
        for e1, e2 in zip(s1, s2):
            if e1 not in d:
                d[e1] = {e2: 1}
            else:
                if e2 not in d[e1]:
                    d[e1][e2] = 1
                else:
                    d[e1][e2] += 1
        return d
    d1 = scan(s1, s2)
    d2 = scan(s2, s1)

    to_one_k1 = [k for k, v in d1.items() if len(v.keys())==1]
    # one_to_k2 = [k for k, v in d2.items() if len(v.keys())==1]
    one_to_k1 = [list(v.keys())[0] for k, v in d2.items() if len(v.keys())==1]    
    one_to_one_k1 = set(to_one_k1) & set(one_to_k1)
    one_to_many_k1 = set(one_to_k1) - set(to_one_k1)
    many_to_one_k1 = set(to_one_k1) - set(one_to_k1)
    many_to_many_k1 = d1.keys() - one_to_one_k1 - one_to_many_k1 - many_to_one_k1
    
    if verbose:
        print(f'1-1 {len(one_to_one_k1)} {len(one_to_one_k1) / len(set(d1.keys())) *100:.0f}%, 1-m {len(one_to_many_k1)} {len(one_to_many_k1) / len(set(d1.keys())) *100:.0f}%, m-1 {len(many_to_one_k1)} {len(many_to_one_k1) / len(set(d1.keys())) *100:.0f}%, m-m {len(many_to_many_k1)} {len(many_to_many_k1) / len(set(d1.keys())) *100:.0f}%, total {len(set(d1.keys()))}')
    
    return {'count_k1': {'total': len(set(d1.keys())),
                         # 'total_k2': len(set(d2.keys())),
                         '1-1': len(one_to_one_k1),
                         '1-m': len(one_to_many_k1),
                         'm-1': len(many_to_one_k1),
                         'm-m': len(many_to_many_k1)},
           'k1': {'1-1': one_to_one_k1,
                  '1-m': one_to_many_k1,
                  'm-1': many_to_one_k1,
                  'm-m': many_to_many_k1}}