#https://stackoverflow.com/questions/14059094/i-want-to-multiply-two-columns-in-a-pandas-dataframe-and-add-the-result-into-a-n
#https://stackoverflow.com/questions/27975069/how-to-filter-rows-containing-a-string-pattern-from-a-pandas-dataframe
#https://chrisalbon.com/python/pandas_dataframe_importing_csv.html
#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.quantile.html
#https://stackoverflow.com/questions/18079563/finding-the-intersection-between-two-series-in-pandas

# distribution of statistics by experience/country

# ability to filter...
import sys
import argparse
from os.path import isfile
import pandas as pd
import numpy as np
import logging

logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.INFO)

def verify_file(arg):
    if not isfile(arg):
        raise argparse.ArgumentTypeError("path does not exist: %s" % arg)
    return arg

def parse_assignments(assignments):
    """
    Parses assignments like... value>=0.1  or  value=0.5
    Returns a list of 3 element tuples (name, operator, value)
    """
    if not isinstance(assignments, (list, tuple, set)):
        print "GOT!!! %s" % assignments
        raise ValueError("expected sequence, got: %s" % type(assignments))
    
    values = []

    for var in assignments:
        operator = None
        if '>=' in var:
            operator = '>='
        elif '<=' in var:
            operator = '<='
        elif '<' in var:
            operator = '<'
        elif '>' in var:
            operator = '>'
        elif '=' in var:
            operator = '='
        else:
            raise ValueError('assignment is missing (=, >=, <=) in value: %s' % var)
        
        name, value = var.split(operator)
        # support single-quotes in arguments 
        value.replace("'", '"')

        values.append((name.strip(), operator, value.strip()))

    return values

def get_quantile_statements(dataframe, quantile_values):
    statements = []
    for column, operator, percent in quantile_values:
        try:
            percent = float(percent)
        except ValueError:
            raise argparse.ArgumentTypeError('invalid quantile value: %s (not a decimal)' % value)
        quantile = dataframe[column].quantile(percent)
        statements.append('{column}{operator}{quantile}'.format(column=column,operator=operator,quantile=quantile))

    return statements

def get_aggregate_dict(aggregate_values):
    """
    Generate a aggregate dictionary that can be passed to pandas .agg() 

    Example::
        {'Users': np.sum, 'apply_gt1': np.mean, 'click_gt1': np.mean}
    """
    aggs = {}
    for  column, _, numpy_func in aggregate_values:
        agg_fn = getattr(np, numpy_func, None)
        if agg_fn is None:
            raise ValueError('%s is not a valid numpy aggregate function!' % numpy_func)
        aggs[column] = agg_fn

    return aggs

        
def and_pandas_querystr(querystrings):
    def strip_parenths(qs):
        qs = qs.strip()
        if qs[0] == '(' and qs[-1] == ')':
            qs = qs[1:-1]
        return qs

    querystrings = [strip_parenths(qs) for qs in querystrings]
    query = '(' + ') & ('.join(querystrings) + ')'
    # If the generated query is empty, generate a query that is true for all rows.
    if len(query) == 2:
        query = '(count>=1)'
    return query



def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=verify_file, help='data file')
    parser.add_argument('-hr', '--header', required=False, help='header-row (1 index)', default=None)
    parser.add_argument('-g', '--groupby', required=False, nargs='*', default=('SearchExperience', 'Country'), help='group by field names')
    parser.add_argument('-a', '--aggregate', required=False, nargs='*', default=parse_assignments(('count=sum', 'apply_ratio=mean', 'search_ratio=mean', 'apply_result_ratio=mean', 'binary_apply=mean', 'binary_search=mean')), type=parse_assignments, help='show aggregates (sum, mean)')
    parser.add_argument('-f', '--filter', required=False, nargs='*', default=tuple(['Country=="United States"']), help='Filters, any valid pandas query expression, multiple filters will be AND\'d')
    parser.add_argument('-q', '--quantile', required=False, nargs='*', default=parse_assignments(['Search<=0.95', 'ApplyClick<=0.95']), type=parse_assignments, help="Quantile filter, the quantile is found for the entire data-set, it ignores filters")
    parser.add_argument('--show-original', required=False, action='store_true', default=False, help="Show a grouping of the original data, without filtering first")
    parser.add_argument('--min-count', required=False, type=int, default=None, help="Minimum count in groupby result to show group")
    parser.add_argument('--write-csv', required=False, default=None, help="output csv")
    args = parser.parse_args(argv[1:])


    # Load a Pandas data-frame from csv
    df = pd.read_csv(args.csv, header=(args.header or 'infer'))

    # check for spaces in column names, this tool doesn't support them, so we'll remove them for you!
    col_spaces = [' ' in col for col in df.columns.values]
    if col_spaces:
        print "\n!!! Some of your columns contained whitespace, we've renamed the columns to eliminate whitespace. This tool does not work with whitespace column names!"
        df.columns = [col.replace(' ', '') for col in df.columns.values]

    # Add some columns with calculated data.
    df['apply_ratio'] = df.ApplyClick / df.Search
    df['search_ratio'] = df.SearchResultClick / df.Search
    df['apply_result_ratio'] = df.ApplyClick / df.SearchResultClick 
    df['binary_search'] = df.Search >= 1
    df['binary_click'] = df.SearchResultClick >= 1
    df['apply_lt_search'] = df.ApplyClick < df.Search
    df['apply_lt_click'] = df.ApplyClick < df.SearchResultClick
    df['search_gt_click'] = df.Search > df.SearchResultClick
    df['binary_apply'] = df.ApplyClick >= 1
    df['count'] = 1


    print "\nDATA COLUMNS:"
    print "-----------------------------"
    print '    ' + '\n    '.join(df.columns.values)
    print ''

    # replace inf, nan with 0
    df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    # validate columns used for filtering
    columns = set(list(args.groupby) + [v[0] for v in args.quantile])
    invalid = set(columns) - set(df.columns.values)
    if invalid:
        raise ValueError('some columns referenced in arguments dont exist in the data, INVALID: %s' % list(invalid))

    pd_quantiles = get_quantile_statements(df, args.quantile)
    pd_aggregates = get_aggregate_dict(args.aggregate)
    pd_filters = args.filter

    pd_query = and_pandas_querystr(list(pd_quantiles) + list(pd_filters))

    log.info('QUANTILES: %s' % pd_quantiles)
    log.info('AGGREGATES: %s' % pd_aggregates)
    log.info('FILTERS: %s' % pd_filters)
    log.info('FULL QUERY: %s' % pd_query)

    # SHOW THE FULL DATA GROUPBY
    if args.show_original:
        grp = df.groupby(args.groupby).agg(pd_aggregates)
        if args.min_count:
            grp = grp.query("count >= @args.min_count")
        print "\nGROUPBY - ALL DATA ..."
        print "-"*60
        print grp

    # SHOW THE FILTERED DATA GROUP BY
    grp = df.query(pd_query).groupby(args.groupby).agg(pd_aggregates)
    if args.min_count:
        grp = grp.query("count >= @args.min_count")
    print "\nGROUPBY - FILTERED, AGGREGATE, GROUPBY"
    print "-"*60
    print grp

    if args.write_csv:
        # see: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_csv.html
        grp.to_csv(args.write_csv, header=True)

if __name__ == '__main__':
    sys.exit(main(sys.argv))