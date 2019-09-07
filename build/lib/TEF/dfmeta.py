import numpy as np
import pandas as pd
import io
import re
import warnings
from scipy.stats import skew, skewtest
from scipy.stats import rankdata

from .plot_1var import *
# from plot_1var import * # for local testing only

from IPython.display import HTML

def print_list(l, br=', '):
    o = ''
    for e in l:
        o += str(e) + br
    return o[:-len(br)]

def summary(s, max_lev=10, br_way=', ', sum_num_like_cat_if_nunique_small=5):
    '''
    a function that takes a series and returns a summary string
    '''
    if s.nunique(dropna=False) == 1:
        return(f'all the same: {s.unique()[0]}')
    elif s.notnull().sum() == 0:
        return(f'all are NaNs')

    if s.dtype.name in ['object', 'bool', 'category'] or \
        (('float' in s.dtype.name or 'int' in s.dtype.name) \
            and s.nunique() <= sum_num_like_cat_if_nunique_small):
        if len(s.unique()) <= max_lev:
            # consider drop na?
            vc = s.value_counts(dropna=False, normalize=True)
            # vc = s.value_counts(dropna=True, normalize=True)
            s = ''
            for name, v in zip(vc.index, vc.values):
                s += f'{name} {v*100:>2.0f}%' + br_way
            return s[:-len(br_way)]
        else:
            vc = s.value_counts(dropna=False, normalize=True)
            # vc = s.value_counts(dropna=True, normalize=True)
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
    elif 'float' in s.dtype.name or 'int' in s.dtype.name:
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

def possible_dup_lev(series, threshold=0.9, truncate=False):
    if series.dtype.name not in ['category', 'object']:
        return ''

    try:
        from fuzzywuzzy import fuzz
    except ImportError:
        sys.exit("""Please install fuzzywuzzy first
                    install it using: pip install fuzzywuzzy
                    if installing the dependency python-levenshtein is failed and you are using Anaconda, try
                    conda install -c conda-forge python-levenshtein""")

    threshold *= 100
    l = series.unique().tolist()
    if len(l) > 100: # maybe should adjust
        return ''
    l = [y for y in l if type(y) == str] # remove nan, True, False
    candidate = []
    for i in range(len(l)):
        for j in range(i+1, len(l)):
            if any([fuzz.ratio(l[i], l[j]) > threshold, 
                fuzz.partial_ratio(l[i], l[j]) > threshold,
                fuzz.token_sort_ratio(l[i], l[j]) > threshold, 
                fuzz.token_set_ratio(l[i], l[j]) > threshold]):
                candidate.append((l[i], l[j]))
    o = '; '.join(['('+', '.join(can)+')' for can in candidate])
    if truncate and len(o) > 1000:
        o = o[:1000] + '...truncated call TEF.possible_dup_lev(series) for a full result'
    return o

def dfmeta(df, description=None, max_lev=10, transpose=True, sample=True,
           style=True, color_bg_by_type=True, highlight_nan=0.5, in_cell_next_line=True,
           drop=None,
           check_possible_error=True, dup_lev_prop=0.9,
           fitted_feat_imp=None,
           plot=True,
           standard=False):

    # validation
    assert max_lev > 2, 'max_lev should > 2'
    assert sample < df.shape[0], 'sample should < nrows'
    assert drop is None or 'NaNs' not in drop, 'Cannot drop NaNs for now'
    assert drop is None or 'dtype' not in drop, 'Cannot drop dtype for now'

    warnings.simplefilter('ignore', RuntimeWarning) # caused from skewtest, unknown

    if standard: # overwrite thise args
        check_possible_error = False
        sample = False
        # drop=['unique levs']

    # the first line, shape, dtypes, memory
    buffer = io.StringIO()
    df.info(verbose=False, buf=buffer)
    s = buffer.getvalue()
    if style == False:
        print(f'shape: {df.shape}')
        print(s.split('\n')[-3])
        print(s.split('\n')[-2])
        color_bg_by_type, highlight_nan, in_cell_next_line = False, False, False

    br_way = "<br/> " if in_cell_next_line else ", " # notice a space here
    
    o = pd.DataFrame(columns=df.columns)
    
    o.loc['idx'] = list(range(df.shape[1]))
    o.loc['dtype'] = df.dtypes
    
    if description is not None:
        o.loc['description'] = ''
        for col, des in description.items():
            if col in df.columns.tolist():
                o.loc['description', col] = des
    
    o.loc['NaNs'] = df.apply(lambda x: f'{sum(x.isnull())}{br_way}{sum(x.isnull())/df.shape[0]*100:.0f}%')

    o.loc['unique counts'] = df.apply(lambda x: f'{len(x.unique())}{br_way}{len(x.unique())/df.shape[0]*100:.0f}%')
    
    # def unique_index(s):
    #     if len(s.unique()) <= max_lev:
    #         o = ''
    #         for i in s.value_counts(dropna=False).index.tolist():
    #             o += str(i) + br_way
    #         return o[:-len(br_way)]
    #     else:
    #         return ''
    # o.loc['unique levs'] = df.apply(unique_index, result_type='expand')
    
    o.loc['summary'] = df.apply(summary, result_type='expand', max_lev=max_lev, br_way=br_way) # need result_type='true' or it will all convert to object dtype
    # maybe us args=(arg1, ) or sth?

    if plot and style:
        o.loc['summary plot'] = ['__TO_PLOT_TO_FILL__'] * df.shape[1]

    if fitted_feat_imp is not None:
        def print_fitted_feat_imp(fitted_feat_imp, indices):
            fitted_feat_imp = fitted_feat_imp[fitted_feat_imp.notnull()]
            o = pd.Series(index=indices)
            rank = len(fitted_feat_imp) - rankdata(fitted_feat_imp).astype(int) + 1
            for i in range(len(fitted_feat_imp)):
                o[fitted_feat_imp.index[i]] = f'{rank[i]:.0f}/{len(fitted_feat_imp)} {fitted_feat_imp[i]:.2f} {fitted_feat_imp[i]/sum(fitted_feat_imp)*100:.0f}%'
            o.loc[o.isnull()] = ''
            return o
        o.loc['fitted feature importance'] = print_fitted_feat_imp(fitted_feat_imp, df.columns)

    if check_possible_error:
        def possible_nan(x):
            if x.dtype.name not in ['category', 'object']:
                return ''

            check_list = ['NEED', 'nan', 'Nan', 'nAn', 'naN', 'NAn', 'nAN', 'NaN', 'NAN']
            check_list_re = [' +', '^null$', r'^[^a-zA-Z0-9]*$']
            o = ''
            if sum(x==0) > 0:
                o += f' "0": {sum(x==0)}, {sum(x==0)/df.shape[0]*100:.2f}%{br_way}'
            for to_check in check_list:
                if to_check in x.unique().tolist():
                    o += f' "{to_check}": {sum(x==to_check)}, {sum(x==to_check)/df.shape[0]*100:.2f}%{br_way}'
            for to_check in check_list_re:
                is_match = [re.match(to_check, str(lev), flags=re.IGNORECASE) is not None for lev in x]
                if any(is_match):
                    to_print = ', '.join(x[is_match].unique())
                    o += f' "{to_print}": {sum(is_match)}, {sum(is_match)/df.shape[0]*100:.2f}%{br_way}'
            return o
        o.loc['possible NaNs'] = df.apply(possible_nan)

        o.loc['possible dup lev'] = df.apply(possible_dup_lev, args=(dup_lev_prop, True))

    if sample != False:
        if sample == True and type(sample) is not int:
            sample_df = df.sample(3).sort_index()
        elif sample == 'head':
            sample_df = df.head(3)
        elif type(sample) is int:
            sample_df = df.sample(sample)
        sample_df.index = ['row ' + str(x) for x in sample_df.index.tolist()]
        o = o.append(sample_df)

    if drop:
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
                        'int8': '#fefee7',
                        'int16': '#fefee7',
                        'int32': '#fefee7',
                        'int64': '#fefee7',
                        'uint8': '#fefee7',
                        'uint16': '#fefee7',
                        'uint32': '#fefee7',
                        'uint64': '#fefee7',
                        'float16': '#fef2e7',
                        'float32': '#fef2e7',
                        'float64': '#fef2e7',
                        'bool': '#e7fefe',
                        'category': '#e7ecfe'}
                # if data.iloc[2] not in cmap: # idx 2 is dtype
                if data.loc['dtype'].name not in cmap:
                    cell_rule += "background-color: grey"
                else:
                    cell_rule += "background-color: {}".format(cmap[data.loc['dtype'].name])
                rule = [cell_rule] * len(data)
                if transpose:
                    rule[0] = 'background-color: white;'
            else:
                rule = [''] * len(data)
            
            # if float(data.iloc[3][-3:-1])/100 > highlight_nan or data.iloc[3][-4:] == '100%': # idx 3 is NaNs
            if float(data.loc['NaNs'][-3:-1])/100 > highlight_nan or data.loc['NaNs'][-4:] == '100%':
                rule[np.where(data.index=='NaNs')[0][0]] += '; color: red'

            if data.loc['unique counts'][-4:] == '100%': # all unique
                rule[np.where(data.index=='unique counts')[0][0]] += '; color: blue'
            elif data.loc['unique counts'][:7] == f'1{br_way}': # all the same
                rule[np.where(data.index=='unique counts')[0][0]] += '; color: red' 

            if fitted_feat_imp is not None:
                if data.loc['fitted feature importance'][:2] in ['1/', '2/', '3/']:
                    rule[np.where(data.index=='fitted feature importance')[0][0]] += '; font-weight: bold'
            return rule
        o = o.style.apply(style_rule, axis=int(transpose)) # axis=1 for row-wise, for transpose=True
        if transpose:
            o = o.hide_index()

    if style: # caption
        s = print_list(s.split('\n')[-3:-1], br='; ')
        o = o.set_caption(f"shape: {df.shape}; {s}")

    o = o.render() # convert from pandas.io.formats.style.Styler to html code
    if plot and style:
        for c in range(df.shape[1]):
            html_1var = plot_1var_series(df, c, max_lev, log_numeric=False, save_plt=None, return_html=True)
            o = o.replace('__TO_PLOT_TO_FILL__', html_1var, 1)
    o = HTML(o) # convert from html to IPython.core.display.HTML

    return o

def dfmeta_to_htmlfile(styled_df, filename, head=''):
    '''
    styled_df should be <class 'IPython.core.display.HTML'>
    '''
    r = f'<h1>{head}</h1>\n' + '<body>\n' + styled_df.data + '\n</body>'
    with open(filename, 'w') as f:
        f.write(r)
    return f'{filename} saved'

# def print_html_standard(df, description):
#     meta = dfmeta(df, 
#         description=description,
#         check_possible_error=False, sample=False, drop=['unique levs'])

#     dfmeta_verbose_html = ''
#     buffer = io.StringIO()
#     df.info(verbose=False, buf=buffer)
#     s = buffer.getvalue().split('\n')
#     dfmeta_verbose = f"shape: {df.shape}<br/>{s[-3]}<br/>{s[-2]}"
#     dfmeta_verbose_html = '<p>' + dfmeta_verbose + '</p>'

#     r = dfmeta_verbose_html + '<body>\n' + meta.data + '\n</body>'

#     for e in r.split('\n'):
#         print(e)

# def dfmeta_to_htmlfile_standard(df, description, filename, head):
#     '''
#     a function that call dfmeta and then dfmeta_to_htmlfile using a standard configuration
#     '''
#     meta = dfmeta(df, 
#         description=description,
#         check_possible_error=False, sample=False, drop=['unique levs'])
#     return dfmeta_to_htmlfile(meta, filename, head)

def get_desc_template(df, var_name='desc', suffix_idx=False):
    print(var_name, '= {')
    max_cn = max([len(x) for x in df.columns.tolist()]) + 1
    len_cn = 25 if max_cn > 25 else max_cn
    for i in range(df.shape[1]):
        c = df.columns[i]
        c += '"'
        if c[:-1] != df.columns.tolist()[-1]:
            if suffix_idx == False:
                print(f'    "{c:{len_cn}}: "",')
            else:
                print(f'    "{c:{len_cn}}: "", # {i}')
        else:
            if suffix_idx == False:
                print(f'    "{c:{len_cn}}: ""')
            else:
                print(f'    "{c:{len_cn}}: ""  # {i}')
    print('}')

def get_desc_template_file(df, filename='desc.py', var_name='desc', suffix_idx=False):
    '''%run filename.py'''
    max_cn = max([len(x) for x in df.columns.tolist()]) + 1
    len_cn = 25 if max_cn > 25 else max_cn
    o = var_name + ' = {' + '\n'
    for i in range(df.shape[1]):
        c = df.columns[i]
        c += '"'
        if c[:-1] != df.columns.tolist()[-1]:
            o += f'    "{c:{len_cn}}: "", # {i}' + '\n'
        else:
            o += f'    "{c:{len_cn}}: ""  # {i}' + '\n'
    o += '}'

    with open(filename, 'w') as f:
        f.write(o)
    return f'{filename} saved'