import numpy as np
import pandas as pd
import re
import io

def convert_column_list(columns, idx_or_labels, to_idx_only=False, to_label_only=False):
    # assert sum([idx is not None, labels is not None]) == 1, 'should only provide either idx or labels'
    # if idx is not None:
    #     return [columns[i] for i in idx]
    # elif labels is not None:
    #     return [np.where(columns==cn)[0][0] for cn in labels]
    o = []
    for i in idx_or_labels:
        # int -> str
        if isinstance(i, int):
            if not to_idx_only:
                o.append(columns[i])
            else:
                o.append(i)
        # str -> int
        elif isinstance(i, str):
            if not to_label_only:
                o.append(np.where(columns==i)[0][0])
            else:
                o.append(i)
    return o

def auto_set_dtypes(df, max_num_lev=10, 
                    set_datetime=[], set_category=[], set_int=[], set_float=[], set_object=[], set_bool=[],
                    set_datetime_by_pattern=r'\d{4}-\d{2}-\d{2}',
                    verbose=1):
    df = df.copy() # need this or will change original df
    if verbose:
        record = pd.DataFrame({'before': df.dtypes}).transpose()

        buffer = io.StringIO()
        df.info(verbose=False, buf=buffer)
        s = buffer.getvalue()
        dtypes_before = s.split('\n')[-3]

    for c in range(df.shape[1]): # for every cols
        cur = df.iloc[:, c]

        if c in set_object or c in convert_column_list(df.columns, set_object, to_idx_only=True):
            df.iloc[:, c] = df.iloc[:, c].astype('object') # don't know why if its already datetime it doesn't change
        elif c in set_datetime or c in convert_column_list(df.columns, set_datetime, to_idx_only=True):
            df.iloc[:, c] = pd.to_datetime(cur, errors='coerce')
        elif c in set_bool or c in convert_column_list(df.columns, set_bool, to_idx_only=True):
            df.iloc[:, c] = cur.astype('bool')
        elif c in set_category or c in convert_column_list(df.columns, set_category, to_idx_only=True):
            df.iloc[:, c] = cur.astype('category')
        elif c in set_int or c in convert_column_list(df.columns, set_int, to_idx_only=True):
            df.iloc[:, c] = cur.astype('Int64') # use 'Int64' instead of int to ignore nan, can't handle object type
        elif c in set_float or c in convert_column_list(df.columns, set_float, to_idx_only=True):
            df.iloc[:, c] = cur.astype(float)

        else:
            if set_datetime_by_pattern:
                if sum(cur.notnull()) > 0: # not all are null
                    fisrt_possible_date = cur[cur.notnull()].iloc[0] # use the first not null
                    if re.match(set_datetime_by_pattern, str(fisrt_possible_date)):
                        df.iloc[:, c] = pd.to_datetime(cur, errors='coerce')
                        continue
            if set(cur.unique().tolist()) == set([False, True]):
                # 0, 1 will becomes bool here
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
            record.loc['idx'] = list(range(df.shape[1]))
            record = record.append(df.sample(3), sort=False)
            pd.set_option('display.max_columns', record.shape[1])
            print(record)

        # check possible id cols
        if verbose != 0:
            l = df.columns.tolist()
            check_list = ['id', 'key', 'number']
            ignore_list = ['bid', 'accident']
            possible_list = [c for c in range(len(l)) if 
                any([x in l[c].lower() for x in check_list]) and
                all([x not in l[c].lower() for x in ignore_list]) and 
                df.iloc[:, c].dtype.name != 'object' and  # ignore current object
                df.iloc[:, c].nunique() / df.shape[0] > 0.5] # number of unique should be high enough
            if len(possible_list) > 0:
                print()
                print(f'possible identifier cols: {", ".join([str(c)+" "+l[c] for c in possible_list])}')
                print(f'consider using set_object={possible_list}')

    return df