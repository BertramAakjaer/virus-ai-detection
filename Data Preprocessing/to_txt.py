import pandas as pd

# Create an empty DataFrame with defined columns
df = pd.DataFrame(columns=['Numerical Feature', 'Dictionaries', 'Label'])

# Now you can add rows as needed using loc
df.loc[len(df)] = [10, [{"WSOCK32.dll": ["bind", "listen"]}, {"API": ["CreateProcess", "ReadFile"]}], 'Malicious']
df.loc[len(df)] = [5, [{"KERNEL32.dll": ["read", "write"]}], 'Benign']
df.loc[len(df)] = [15, [{"USER32.dll": ["message", "input"]}, {"API": ["SendMessage"]}], 'Benign']
df.loc[len(df)] = [20, [{"NTDLL.dll": ["allocate", "free"]}], 'Malicious']

print(df)
