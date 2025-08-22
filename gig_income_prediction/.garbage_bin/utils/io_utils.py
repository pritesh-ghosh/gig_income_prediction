'''

def load_csv(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def save_csv(dataframe, file_path):
    import pandas as pd
    dataframe.to_csv(file_path, index=False)

def load_json(file_path):
    import json
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path):
    import json
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def load_pickle(file_path):
    import pandas as pd
    return pd.read_pickle(file_path)

def save_pickle(data, file_path):
    import pandas as pd
    pd.to_pickle(data, file_path)
    '''