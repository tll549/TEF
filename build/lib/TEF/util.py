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
            print([df2.columns[c] for c in no_change])
    return df2

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