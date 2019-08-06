def plot_1var_by_cat_y(df, y, max_num_lev=20, log_numeric=True,
    kind_for_num='boxen'):
    '''
    '''
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