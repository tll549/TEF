# Todo for TEF
## auto_set_dtypes

- check [True, False, nan]
- figure out what is T in datetime
- let r'd{4}-d{2}-d{2}[ T]d{2}:d{2}:d{2}' becomes time part is optional
- fill ' ' with nan
- check similar strings in object or category type
- check if needs to take log
- consider those arguments and functions in descripe, summary, dtypes

## dfmeta

- have warning when the first time run
- check package imported or not
- set display.max_columns back, also set max_row (cant)
- display 0 for true 0, 0.00% for not true 0 (can't becuase need it in summary)
- possible nans should also check lower
- why sometime can calculate log skew
- detect if one col's level is contained in another col, like main_reason and detailed_reason
- check ., ' ', '  ', characters
- check strings using FuzzyWuzzy: https://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
- make it a class, have method to get dict template, get html code, save html page

## plot_1var

- save and print a specific plot by col name
- for numeric, auto detect should take log or not, or just plot another log plot
    (maybe) for category, handle too many levels like object
    
## plot_1var_by_cat_y

- percentage for heatmap
- for numeric, auto detect should take log or not, or just plot another log plot
    (maybe) for category, handle too many levels like object