import pandas as pd
pd.set_option('display.max_columns', None)

def is_number(s):
    if s is None:
        s = np.nan

    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def read_data_cn(fname, interval, forecast):
    """
    interval: interval of group data, days
    forecast: whether the patient will die within the next forecast hours
    """

    # read data
    data_df = pd.read_excel(fname, encoding='gbk', index_col=[0])
    # group by interval
    data_df['total_period'] = data_df['出院时间'] - data_df['入院时间']
    data_df['total_period'] = data_df['total_period'].apply(lambda x: x.days//interval)
    data_df['period'] = data_df['出院时间'] - data_df['RE_DATE']
    data_df['period'] = data_df['period'].apply(lambda x: x.days//interval)
    data_df['period'] = data_df['total_period'] - data_df['period']
    data_df = data_df.groupby(['PATIENT_ID', 'period']).last()
    # make outcome
    decease = data_df['出院方式'].values # whether the patient will die in hospital
    data_df['decease'] = decease
    decompensation = data_df['出院时间'] - data_df['RE_DATE'] # whether the patient will die within the next 24 hours
    decompensation = decompensation.apply(lambda x: int(x.total_seconds()/3600 < forecast))
    data_df['decompensation'] = data_df['decease'] * decompensation
    # make it cleaner
    data_df = data_df.drop(['RE_DATE', '出院方式', '入院时间', '出院时间', 'total_period'], axis=1)
    data_df = data_df.applymap(lambda x: x.replace('>', '').replace('<', '') if isinstance(x, str) else x)
    data_df = data_df.applymap(lambda x: x if is_number(x) else -1)
    data_df = data_df.astype(float)
    columns = ['decease', 'decompensation', '乳酸脱氢酶', '超敏C反应蛋白', '淋巴细胞(%)']
    data_df = data_df[columns]
    return data_df

if __name__ == "__main__":

    interval = 3
    forecast = 72
    df_train = read_data_cn('data/time_series_375_prerpocess.xlsx', interval, forecast)
    df_test = read_data_cn('data/time_series_test_110_preprocess.xlsx', interval, forecast)
    
    
    
    
    
