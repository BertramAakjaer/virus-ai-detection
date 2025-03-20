import pandas as pd

def init_table():
    return pd.DataFrame(columns=['SizeOfCode', 'Dictionaries', 'Label'])

def add_row(df, numerical_feature, dictionaries, label):
    df.loc[len(df)] = [numerical_feature, dictionaries, label]

if __name__ == '__main__':
    df = init_table()
    
    add_row(df, 10, {"WSOCK32.dll": ["bind", "listen"],"API": ["CreateProcess", "ReadFile"]}, 'Malicious')
    
    print(df)
    
