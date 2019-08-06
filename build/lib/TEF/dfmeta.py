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