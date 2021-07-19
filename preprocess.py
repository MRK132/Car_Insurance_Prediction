import pandas as pd

def preprocess(df):
    """
    Preprocessing steps from first portion of notebook.
    
    Args:
        df: Input dataframe
        
    Returns:
        df: Dataframe after preprocessing
    
    """
    # Create a mapper for education 

    scale_mapper = {
        "primary": 0,
        "secondary": 1,
        "tertiary": 2,
    }

    df['Education'] = df['Education'].replace(scale_mapper)

    # Use median to impute:
    df['Education'].fillna(df['Education'].median(), inplace=True)

    # one-hot encode the categorical columns with NaNs
    df = pd.get_dummies(df, columns=['Job', 'Communication', 'Outcome'], 
                            dummy_na=True, 
                            drop_first=False)

    # convert time features to two new features = call duration,
    # and morning,afternoon,evening bins => Period_of_day_call

    start_secs=[]
    for time in df['CallStart']:
        td = pd.Timedelta(time)
        time = td.seconds
        start_secs.append(time)

    end_secs=[]
    for time in df['CallEnd']:
        td = pd.Timedelta(time)
        time = td.seconds
        end_secs.append(time)


    call_duration=[]
    for start,end in zip(pd.Series(start_secs), pd.Series(end_secs)):
        duration = end-start
        call_duration.append(duration)

    df['Call_duration '] = call_duration
    df['Start_call_secs'] = start_secs

    time=df['Start_call_secs']


    period_of_day_call = []

    for call_time in time:                                                         # 6 potential bins
        if pd.Timedelta('6:00:00').seconds < call_time <= pd.Timedelta('09:00:00').seconds:
            period_of_day_call.append('6am-9am')
        elif pd.Timedelta('9:00:00').seconds < call_time <= pd.Timedelta('12:00:00').seconds:
            period_of_day_call.append('9am-12am')
        elif pd.Timedelta('12:00:00').seconds < call_time <= pd.Timedelta('15:00:00').seconds:
            period_of_day_call.append('12am-3pm')
        elif pd.Timedelta('15:00:00').seconds < call_time <= pd.Timedelta('18:00:00').seconds:
            period_of_day_call.append('3pm-6pm')
        elif pd.Timedelta('18:00:00').seconds < call_time <= pd.Timedelta('21:00:00').seconds:
            period_of_day_call.append('6pm-9pm')
        elif pd.Timedelta('21:00:00').seconds < call_time <= pd.Timedelta('23:59:00').seconds:
            period_of_day_call.append('9pm-12pm')

    df['Period_of_day_call'] = period_of_day_call

    df = df.drop(columns=['CallStart', 'CallEnd', 'Start_call_secs'])

    # now one-hot encode all other categorical columns
    df = pd.get_dummies(df, drop_first=False)

    return df