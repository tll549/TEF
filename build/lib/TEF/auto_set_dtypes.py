def auto_set_dtypes(df, max_num_lev=10, 
                    set_datetime=[], set_category=[], set_int=[], set_object=[], set_bool=[],
                    set_datetime_by_pattern=r'\d{4}-\d{2}-\d{2}',
                    verbose=1):
    '''

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
            # print(set(cur.unique().tolist()), set([False, True]), set(cur.unique().tolist()) == set([False, True]))
            if set(cur.unique().tolist()) == set([False, True]):
                df.iloc[:, c] = cur.astype('bool')
            elif len(cur.unique()) <= max_num_lev and cur.dtype.name == 'object': # only change to category from object, in case changing for example int (2, 3, 4, 5) to category
                df.iloc[:, c] = cur.astype('category')
                
    if verbose:
        if verbose == 'summary' or verbose == 1:
            buffer = io.StringIO()
            df.info(verbose=False, buf=buffer)
            s = buffer.getvalue()
            dtypes_after = s.split('\n')[-3]
            print('before', dtypes_before)
            print('after ', dtypes_after)
        elif verbose == 'detailed' or verbose >= 2:
            record = record.append(pd.DataFrame({'after': df.dtypes}).transpose(), sort=False)
            record = record.append(df.sample(3), sort=False)
            pd.set_option('display.max_columns', record.shape[1])
            print(record)

        # check possible id cols
        if verbose != 0:
            l = df.columns.tolist()
            possible_list = [c for c in range(len(l)) if 'id' in l[c].lower()]
            if len(possible_list) > 0:
                print()
                print(f'possible identifier cols: {", ".join([str(c)+" "+l[c] for c in possible_list])}')
                print(f'consider using set_object={possible_list}')

    return df