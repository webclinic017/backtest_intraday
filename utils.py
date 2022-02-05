import os


def save_create_csv(dir_name, file_name, df):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    fullname = f'{os.path.join(dir_name, file_name)}.csv'
    df.to_csv(fullname, index=False)