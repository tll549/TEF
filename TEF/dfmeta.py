def print_list(l, br=', '):
    o = ''
    for e in l:
        o += str(e) + br
    return o[:-len(br)]

def summary(s, max_lev=10, br_way=', '):
    '''
    a function that takes a series and returns a summary string
    '''
    import numpy as np
    from scipy.stats import skew, skewtest

    if s.nunique() == 1:
        return(f'all the same: {s.unique()[0]}')
    elif s.notnull().sum() == 0:
        return(f'all are NaNs')
    if s.dtype.name in ['object', 'bool', 'category']:
        if len(s.unique()) <= max_lev:
            vc = s.value_counts(dropna=False, normalize=True)
            s = ''
            for name, v in zip(vc.index, vc.values):
                s += f'{name} {v*100:>2.0f}%' + br_way
            return s[:-len(br_way)]
        else:
            vc = s.value_counts(dropna=False, normalize=True)
            s = ''
            i = 0
            cur_sum_perc = 0
            for name, v in zip(vc.index, vc.values):
                if i == max_lev or \
                    (i >= 5 and cur_sum_perc >= 0.8) or \
                    (i == 0 and cur_sum_perc < 0.05):
                    # break if the it has describe 80% of the data, or the
                    break
                s += f'{name} {v*100:>2.0f}%' + br_way
                i += 1
                cur_sum_perc += v
            s += f'other {(1-cur_sum_perc)*100:>2.0f}%'
            # return s[:-len(br_way)]
            return s
    elif s.dtype.name in ['float64', 'int64']:
        qs = s.quantile(q=[0, 0.25, 0.5, 0.75, 1]).values.tolist()
        cv = round(s.std()/s.mean(), 2) if s.mean() != 0 else 'nan'
        sk = round(skew(s[s.notnull()]), 2) if len(s[s.notnull()]) > 0 else 'nan'
        o = f'{qs}{br_way}\
            mean: {s.mean():.2f} std: {s.std():.2f}{br_way}\
            cv: {cv} skew: {sk}'
        if sum(s.notnull()) > 8: # requirement of skewtest
            p = skewtest(s[s.notnull()]).pvalue
            o += f'*' if p <= 0.05 else ''
            if min(s[s!=0]) > 0 and len(s[s!=0]) > 8: # take log
                o += f'{br_way}log skew: {skew(np.log(s[s>0])):.2f}'
                p = skewtest(np.log(s[s!=0])).pvalue
                o += f'*' if p != p and p <= 0.05 else ''
        return o
    elif 'datetime' in s.dtype.name:
        qs = s.quantile(q=[0, 0.25, 0.5, 0.75, 1]).values
        dt_range = (qs[-1]-qs[0]).astype('timedelta64[D]')
        if dt_range > np.timedelta64(1, 'D'):
            to_print = [np.datetime_as_string(q, unit='D') for q in qs]
        else:
            to_print = [np.datetime_as_string(q, unit='s') for q in qs]
        return print_list(to_print, br=br_way)
    else:
        return ''


def dfmeta(df, max_lev=10, transpose=True, sample=True, description=None,
           style=True, color_bg_by_type=True, highlight_nan=0.5, in_cell_next_line=True,
           verbose=True, drop=None,
           check_possible_error=True, dup_lev_prop=0.9):
    '''

    '''
    import numpy as np
    import pandas as pd
    import io
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
    # if in_cell_next_line == True:
    #     br_way = "<br/> "
    # elif in_cell_next_line == '<br></br> ': 
    #     br_way = in_cell_next_line
    # else:
    #     br_way = ", "
    br_way = "<br/> " if in_cell_next_line else ", " # notice a space here
    
    o.loc['NaNs'] = df.apply(lambda x: f'{sum(x.isnull())}{br_way}{sum(x.isnull())/df.shape[0]*100:.0f}%')

    o.loc['unique counts'] = df.apply(lambda x: f'{len(x.unique())}{br_way}{len(x.unique())/df.shape[0]*100:.0f}%')
    
    def unique_index(s):
        if len(s.unique()) <= max_lev:
            o = ''
            for i in s.value_counts(dropna=False).index.tolist():
                o += str(i) + br_way
            return o[:-len(br_way)]
        else:
            return ''
    o.loc['unique levs'] = df.apply(unique_index, result_type='expand')
    
    o.loc['summary'] = df.apply(summary, result_type='expand', max_lev=max_lev, br_way=br_way) # need result_type='true' or it will all convert to object dtype
    # maybe us args=(arg1, ) or sth?

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

        def possible_dup_lev(x, threshold):
            if x.dtype.name not in ['category', 'object']:
                return ''
            # l = x.unique().tolist()
            # if len(l) > 100: # maybe should adjust
            #     return ''
            # l = [y for y in l if type(y) == str] # remove nan, True, False
            # candidate = []
            # for i in range(len(l)):
            #     for j in range(i+1, len(l)):
            #         if l[i].lower() in l[j].lower() or l[j].lower() in l[i].lower():
            #             p = min(len(l[i]), len(l[j]))/max(len(l[i]), len(l[j]))
            #             if p >= threshold:
            #                 candidate.append((l[i], l[j]))
            # return '; '.join(['('+', '.join(can)+')' for can in candidate])

            try:
                from fuzzywuzzy import fuzz
            except ImportError:
                sys.exit("""Please install fuzzywuzzy first
                            install it using: pip install fuzzywuzzy
                            if installing the dependency python-levenshtein is failed and you are using Anaconda, try
                            conda install -c conda-forge python-levenshtein""")

            threshold *= 100
            l = x.unique().tolist()
            if len(l) > 100: # maybe should adjust
                return ''
            l = [y for y in l if type(y) == str] # remove nan, True, False
            candidate = []
            for i in range(len(l)):
                for j in range(i+1, len(l)):
                    if fuzz.ratio(l[i], l[j]) > threshold or fuzz.partial_ratio(l[i], l[j]) > threshold  or \
                         fuzz.token_sort_ratio(l[i], l[j]) > threshold  or fuzz.token_set_ratio(l[i], l[j]) > threshold:
                        # print(l[i], l[j], fuzz.ratio(l[i], l[j]), fuzz.partial_ratio(l[i], l[j]), \
                        #     fuzz.token_sort_ratio(l[i], l[j]), fuzz.token_set_ratio(l[i], l[j]))
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

    return o

def dfmeta_to_htmlfile(styled_df, filename, head, original_df=None):
    import io

    dfmeta_verbose_html = ''
    if original_df is not None:
        buffer = io.StringIO()
        original_df.info(verbose=False, buf=buffer)
        s = buffer.getvalue().split('\n')
        dfmeta_verbose = f"shape: {original_df.shape}<br>{s[-3]}<br>{s[-2]}"
        dfmeta_verbose_html = '<p>' + dfmeta_verbose + '</p>'

    r = f'<h1>{head}</h1>\n' + dfmeta_verbose_html + '<body>\n' + styled_df.render() + '\n</body>'
    with open(filename, 'w') as f:
        f.write(r)
    return f'{filename} saved'

def print_html_standard(df, description):
    import io

    meta = dfmeta(df, 
        description=description,
        check_possible_error=False, sample=False, verbose=False, drop=['unique levs'])

    dfmeta_verbose_html = ''
    buffer = io.StringIO()
    df.info(verbose=False, buf=buffer)
    s = buffer.getvalue().split('\n')
    dfmeta_verbose = f"shape: {df.shape}<br/>{s[-3]}<br/>{s[-2]}"
    dfmeta_verbose_html = '<p>' + dfmeta_verbose + '</p>'

    r = dfmeta_verbose_html + '<body>\n' + meta.render() + '\n</body>'

    for e in r.split('\n'):
        print(e)

def save_html_standard(df, description, filename, head):
    '''
    a function that call dfmeta and then dfmeta_to_htmlfile using a standard configuration
    '''
    meta = dfmeta(df, 
        description=description,
        check_possible_error=False, sample=False, verbose=False, drop=['unique levs'])
    return dfmeta_to_htmlfile(meta, filename, head, df)

def get_desc_template(df):
    print('desc = {')
    max_cn = max([len(x) for x in df.columns.tolist()]) + 1
    len_cn = 25 if max_cn > 25 else max_cn
    for c in df.columns.tolist():
        c += '"'
        if c[:-1] != df.columns.tolist()[-1]:
            print(f'    "{c:{len_cn}}: "",')
        else:
            print(f'    "{c:{len_cn}}: ""')
    print('}')

def get_desc_template_file(df, filename='desc.py'):
    '''%run filename.py'''
    max_cn = max([len(x) for x in df.columns.tolist()]) + 1
    len_cn = 25 if max_cn > 25 else max_cn
    o = 'desc = {' + '\n'
    for c in df.columns.tolist():
        c += '"'
        if c[:-1] != df.columns.tolist()[-1]:
            o += f'    "{c:{len_cn}}: "",' + '\n'
        else:
            o += f'    "{c:{len_cn}}: ""' + '\n'
    o += '}'

    with open(filename, 'w') as f:
        f.write(o)
    return f'{filename} saved'