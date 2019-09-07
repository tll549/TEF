# Todo for TEF

- add dependencies on readme
- auto install dependencies
- ct didn't include s1 name

## auto_set_dtypes

- figure out what is T in datetime
- let r'd{4}-d{2}-d{2}[ T]d{2}:d{2}:d{2}' becomes time part is optional
- detect [0, 1], suggest as bool? does it change it to bool now?



## dfmeta

- check same pattern (same number of missing value, nested)
  - detect nested, if one col's level is contained in another col, like main_reason and detailed_reason
- separate return meta dataframe and style, save the meta dataframe somewhere first, maybe use a class instead
- summary for quantile, fix it prints long digits, e.g. 0.20833300054073334
- a col by a time, following by a plot for that col, or, plot in another column (within table)
- have warning when the first time run
- check package imported or not
- set display.max_columns back, also set max_row (cant)
- why sometime can calculate log skew
- make it a class, have method to get dict template, get html code, save html page



## plot_1var

- save and print a specific plot by col name
- for numeric, auto detect should take log or not, or just plot another log plot
    (maybe) for category, handle too many levels like object



## plot_1var_by_cat_y

- handle datatime
- percentage for heatmap
- for numeric, auto detect should take log or not, or just plot another log plot
    (maybe) for category, handle too many levels like object



## fit

- linearSVC should normalize
- consider use other impute method



## others

- find high frequency (bigram? TFIDF?) for object (free-text columns)
- plot_1var_by_num_y
- a function to automatically find relationship between vars, such as corr heatmap that auto selects high corr, handle high dim data
- convert fuzzy numbers to numbers
- convert all cols given by indices to one type, e.g. TEF.astype(object=[], float=[])