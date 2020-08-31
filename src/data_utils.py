from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def get_multiple_choice_value_count(df, field_name, do_norm=False, set_index=False,
                                    filter_condition=None, show_plot=True):
    '''

    :param df:
    :param field_name,
    :param do_norm
    :param set_index,
    :param filter_condition:
    :param show_plot:
    :return:
    '''

    choice_count = {'nan': 0}

    def map_str(sl):
        if isinstance(sl, str):
            for s in sl.split(';'):
                if choice_count.get(s, None):
                    choice_count[s] += 1
                else:
                    choice_count[s] = 1
        else:
            if np.isfinite(sl):
                if choice_count.get(sl, None):
                    choice_count[sl] += 1
                else:
                    choice_count[sl] = 1
            else:
                choice_count['nan'] += 1

    if filter_condition:
        dff = df[df[list(filter_condition.keys())[0]] == list(filter_condition.values())[0]]
    else:
        dff = df.copy()

    dff[field_name].apply(lambda x: map_str(x))
    if not choice_count['nan']:
        choice_count.pop('nan', None)
    choice_count = pd.DataFrame({'MultiLabels': list(choice_count.keys()), 'MultiValues': list(choice_count.values())})
    if do_norm:
        choice_count.MultiValues = choice_count.MultiValues.apply(lambda x: np.around(x / dff.shape[0], 3))

    if show_plot:
        choice_count.plot.bar(x='MultiLabels', y='MultiValues',
                              title=''.join(['chosen values count for:', field_name]),
                              figsize=(20, 10))
    choice_count.rename(columns={'MultiLabels': field_name}, inplace=True)
    choice_count.sort_values(by=field_name, inplace=True)
    if set_index:
        choice_count.set_index(field_name, inplace=True)

    return choice_count


def compare_multiple_choice_with_filter(df, filed_name, filter_field, filter_values=[None, None]):
    '''

    :param df:
    :param filed_name:
    :param filter_field:
    :param filter_values:
    :return:
    '''
    fc_1 = get_multiple_choice_value_count(df, filed_name, do_norm=True, set_index=True,
                                           filter_condition={filter_field: filter_values[1]}, show_plot=False);
    fc_0 = get_multiple_choice_value_count(df, filed_name, do_norm=True, set_index=True,
                                           filter_condition={filter_field: filter_values[0]}, show_plot=False);
    comp_df = pd.merge(fc_1, fc_0, left_index=True, right_index=True)
    ff_n1 = ''.join([filter_field, '_', str(filter_values[1]), '_perc'])
    ff_n0 = ''.join([filter_field, '_', str(filter_values[0]), '_perc'])
    ff_n_diff = ''.join([filter_field, '_Diff_Vals'])
    comp_df.columns = [ff_n1, ff_n0]
    comp_df[ff_n_diff] = comp_df[ff_n1] - comp_df[ff_n0]
    return comp_df


def compare_categorical_dataframes(fc_1, fc_0, fn_1, fn_0):
    '''

    :param fc_1:
    :param fc_0:
    :param fn_1
    :param fn_0
    :return:
    '''
    comp_df = pd.merge(fc_1, fc_0, left_index=True, right_index=True)
    ff_n1 = fn_1
    ff_n0 = fn_0
    ff_n_diff = 'Diff_Vals'
    comp_df.columns = [ff_n1, ff_n0]
    comp_df[ff_n_diff] = comp_df[ff_n1] - comp_df[ff_n0]
    return comp_df


def job_satisfaction(job_sat_level):
    '''

    :param job_sat_level:
    :return:
    '''
    try:
        if job_sat_level == np.nan:
            return 0

        if 'satisfied' == job_sat_level.split()[-1]:
            return 1
        else:
            return 0
    except:
        return 0


def categorize_value_in_range(df, filed_name, cat_number):
    '''

    :param df:
    :param filed_name:
    :param cat_number:
    :return:
    '''
    mean_value = pd.to_numeric(df[filed_name], errors='coerce').mean()
    filed_name_clean = ''.join([filed_name, 'Clean'])
    df[filed_name_clean] = df[filed_name].apply(
        lambda x: np.round(x) if np.isfinite(x) else mean_value)
    cuts = pd.cut(df[filed_name_clean])
    for c in cuts.categories:
        df[filed_name_clean].apply(lambda x: np.round(c.mid) if c.left < x < c.right else x)
    return df[filed_name_clean]


def higher_ed(formal_ed_str):
    '''
    INPUT
        formal_ed_str - a string of one of the values from the Formal Education column

    OUTPUT
        return 1 if the string is  in ("Master's degree", "Doctoral", "Professional degree")
        return 0 otherwise

    '''
    try:
        if formal_ed_str == np.nan:
            return 0
        for s in ("Master’s", "Doctoral", "Professional"):
            if s in formal_ed_str:
                return 1
        else:
            return 0
    except:
        return 0


def get_missing_row_percentage(df):
    """
    # How much data is missing in each row of the dataset?
    :param df:
    :return:
    """

    missing_rows = df.apply(lambda x: round(x.count() / df.shape[1], 2), axis=1)
    df['MissingRows'] = missing_rows

    return df


def get_outliers(df, schema, missing_perc=0.15):
    '''
     Perform an assessment of how much missing data there is in each column of the
     dataset.
    :param df:
    :param schema:
    :params missing_perc:
    :return: dictionary of outliers columns and their percentage of missing value
    '''
    outliers = {}

    for k in schema['Column']:
        nanpr = [y for x, y in df[k].value_counts(dropna=False, normalize=True).items() if str(x) == 'nan']
        if not nanpr:
            continue
        if nanpr[0] < missing_perc:
            continue
        outliers[k] = nanpr[0]
    return pd.DataFrame({'OutlierLabels': list(outliers.keys()), 'OutlierValues': list(outliers.values())})


def get_description(schema, column_name):
    '''
    INPUT - schema - pandas dataframe with the schema of the developers survey
            column_name - string - the name of the column you would like to know about
    OUTPUT -
            desc - string - the description of the column
    '''
    desc = list(schema[schema['Column'] == column_name]['Question'])[0]
    return desc


def find_interval(data_inverval, val):
    '''

    :param data_inverval:
    :param val:
    :return:
    '''
    for c in data_inverval:
        if c.left <= val <= c.right:
            return np.round(c.mid)
    return np.round(val)


def replace_nan_with_mean(x, mean_value):
    '''

    :param x:
    :param mean_value:
    :param replace_nan:
    :return:
    '''
    try:
        if np.isfinite(float(x)):
            return np.round(float(x))
        else:
            return np.round(mean_value)
    except:
        return np.round(mean_value)


def categorize_values_in_range(df, filed_name, cat_number):
    '''
        replace missing values with mean of all values
    :param df:
    :param filed_name:
    :param cat_number:
    :param replace_nan:
    :return:
    '''
    mean_value = pd.to_numeric(df[filed_name], errors='coerce').mean()
    filed_name_clean = ''.join([filed_name, 'Categorized'])
    df[filed_name_clean] = df[filed_name].apply(lambda x: replace_nan_with_mean(x, mean_value))
    cuts = pd.cut(df[filed_name_clean].array, cat_number)
    return df[filed_name_clean].apply(lambda x: find_interval(cuts.categories, x)), filed_name_clean


def coef_weights(coefficients, X_train):
    '''
    INPUT:
    coefficients - the coefficients of the linear model
    X_train - the training data, so the column names can be used
    OUTPUT:
    coefs_df - a dataframe holding the coefficient, estimate, and abs(estimate)

    Provides a dataframe that can be used to understand the most influential coefficients
    in a linear model by providing the coefficient estimates along with the name of the
    variable attached to the coefficient.
    '''
    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = X_train.columns
    coefs_df['coefs'] = lm_model.coef_
    coefs_df['abs_coefs'] = np.abs(lm_model.coef_)
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
    return coefs_df


def impute_with_mean(df, filed_name):
    '''

    :param df:
    :param filed_name:
    :return:
    '''

    try:
        col_mean = df[filed_name].mean()
        new_df = df[filed_name].apply(lambda col: col.fillna(col_mean), axis=0)
    except:
        print('That broke...because column E is a string.')
    return new_df


def create_dummy_df(df, dummy_na=True, cutoff_rate=0.2):
    '''
    INPUT:
    :param: df - pandas dataframe with categorical variables you want to dummy
    :param: dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    :param: cutoff_rate

    OUTPUT:
    df - a new dataframe that has the following characteristics:
            1. contains all columns that were not specified as categorical
            2. removes all the original columns in cat_cols
            3. dummy columns for each of the categorical columns in cat_cols
            4. if dummy_na is True - it also contains dummy columns for the NaN values
            5. Use a prefix of the column name with an underscore (_) for separating
    '''
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        try:
            # for each cat add dummy var, drop original column
            df = pd.concat([df.drop(col, axis=1),
                            pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dummy_na)],
                           axis=1)
        except:
            continue
    for c in df.columns:
        if df[c].mean() < cutoff_rate:
            df.drop(columns=[c], inplace=True)
    return df


def drop_categorical_columns(df):
    '''

    :param df:
    :return:
    '''

    return df.drop(df.select_dtypes(include=['object']).columns, axis=1)


def clean_and_split_data(df, response_col, include_cat_field=False, desired_columns=[], cutoff_rate=0.2, test_size=0.30,
                         rand_state=42):
    '''
    INPUT:
    :param: df - a dataframe holding all the variables of interest
    :param: response_col - a string holding the name of the column
    :param:
    :param: cutoff_rate
    :param: include_cat_field
    :param: test_size
    :param: rand_state

    OUTPUT:

    X, y, X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model

    '''
    # Dropping where the salary has missing values
    df = df.dropna(subset=[response_col], axis=0)

    # Drop columns with all NaN values
    df = df.dropna(how='all', axis=1)

    if desired_columns:
        df = df[desired_columns + [response_col]]

    if include_cat_field:
        df = create_dummy_df(df, True, cutoff_rate)
    else:
        df = drop_categorical_columns(df)

    # Mean function
    fill_mean = lambda col: col.fillna(col.mean())
    # Fill the mean
    df = df.apply(fill_mean, axis=0)

    # Split into explanatory and response variables
    X = df.drop(response_col, axis=1)
    y = df[response_col]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)
    # Split into train and test
    return X, y, x_train, x_test, y_train, y_test


def train_linear_model(x_train, y_train, x_test, y_test):
    '''

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    '''
    lm_model = LinearRegression(normalize=True)  # Instantiate
    lm_model.fit(x_train, y_train)  # Fit

    # Predict using your model
    y_test_preds = lm_model.predict(x_test)
    y_train_preds = lm_model.predict(x_train)

    # Score using your model
    test_score = r2_score(y_test, y_test_preds)
    train_score = r2_score(y_train, y_train_preds)

    return test_score, train_score, lm_model


def find_optimal_lm_mod(X, y, cutoffs, test_size=.30, random_state=42, plot=True):
    '''
    INPUT
    X - pandas dataframe, X matrix
    y - pandas dataframe, response variable
    cutoffs - list of ints, cutoff for number of non-zero values in dummy categorical vars
    test_size - float between 0 and 1, default 0.3, determines the proportion of data as test data
    random_state - int, default 42, controls random state for train_test_split
    plot - boolean, default 0.3, True to plot result

    OUTPUT
    r2_scores_test - list of floats of r2 scores on the test data
    r2_scores_train - list of floats of r2 scores on the train data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''
    r2_scores_test, r2_scores_train, num_feats, results = [], [], [], dict()
    for cutoff in cutoffs:
        # reduce X matrix
        reduce_X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]
        num_feats.append(reduce_X.shape[1])

        # split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size=test_size, random_state=random_state)

        # fit the model and obtain pred response
        lm_model = LinearRegression(normalize=True)
        lm_model.fit(X_train, y_train)
        y_test_preds = lm_model.predict(X_test)
        y_train_preds = lm_model.predict(X_train)

        # append the r2 value from the test set
        r2_scores_test.append(r2_score(y_test, y_test_preds))
        r2_scores_train.append(r2_score(y_train, y_train_preds))
        results[str(cutoff)] = r2_score(y_test, y_test_preds)

    if plot:
        plt.plot(num_feats, r2_scores_test, label="Test", alpha=.5)
        plt.plot(num_feats, r2_scores_train, label="Train", alpha=.5)
        plt.xlabel('Number of Features')
        plt.ylabel('Rsquared')
        plt.title('Rsquared by Number of Features')
        plt.legend(loc=1)
        plt.show()

    best_cutoff = max(results, key=results.get)

    # reduce X matrix
    reduce_X = X.iloc[:, np.where((X.sum() > int(best_cutoff)) == True)[0]]
    num_feats.append(reduce_X.shape[1])

    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size=test_size, random_state=random_state)

    # fit the model
    lm_model = LinearRegression(normalize=True)
    lm_model.fit(X_train, y_train)

    return r2_scores_test, r2_scores_train, lm_model, X_train, X_test, y_train, y_test
