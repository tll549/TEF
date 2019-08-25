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
    import re
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
    import pandas as pd
    c1 = pd.crosstab(s1, s2, margins=True)
    c2 = pd.crosstab(s1, s2, normalize='index')*100
    if col_name is not None:
        c1.columns = col_name + ['All']
        c2.columns = col_name
    o = pd.concat([c1, c2], axis=1, keys=['count', 'proportion'], sort=False)
    o = o[o.index != 'All'] # remove the sum from margins for row, in order to style and sort
    o.columns.names = [None, None]
    if sort:
        if sort == True:
            sort = ('count', 'All')
        o = o.sort_values(sort, ascending=False)
    if head:
        o = o.head(head)
    if style:
        o = o.style.format('{:.0f}').background_gradient(axis=0)
    return o


def load_dataset(name, cache=False, data_home=None, **kws):
    '''
    a function tha totally copied from seaborn
    https://github.com/mwaskom/seaborn/blob/master/seaborn/utils.py
    '''
    import os
    import pandas as pd

    def get_data_home(data_home=None):
        """Return the path of the seaborn data directory.
        This is used by the ``load_dataset`` function.
        If the ``data_home`` argument is not specified, the default location
        is ``~/seaborn-data``.
        Alternatively, a different default location can be specified using the
        environment variable ``SEABORN_DATA``.
        """
        if data_home is None:
            data_home = os.environ.get('data',
                                       os.path.join('~', 'seaborn-data'))
        data_home = os.path.expanduser(data_home)
        if not os.path.exists(data_home):
            os.makedirs(data_home)
        return data_home

    path = ("https://raw.githubusercontent.com/"
            "tll549/TEF/master/TEF/data/{}.csv")
    full_path = path.format(name)
    print(full_path)

    if cache:
        cache_path = os.path.join(get_data_home(data_home),
                                  os.path.basename(full_path))
        if not os.path.exists(cache_path):
            urlretrieve(full_path, cache_path)
        full_path = cache_path

    df = pd.read_csv(full_path, **kws)
    # if df.iloc[-1].isnull().all():
    #     df = df.iloc[:-1]

    # # Set some columns as a categorical type with ordered levels

    # if name == "titanic":
    #     df["class"] = pd.Categorical(df["class"], ["First", "Second", "Third"])
    #     df["deck"] = pd.Categorical(df["deck"], list("ABCDEFG"))

    return df
