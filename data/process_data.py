import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# python3 data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories data from CSV files.

    Args:
        messages_filepath (str): File path to the CSV file containing messages.
        categories_filepath (str): File path to the CSV file containing categories.

    Returns:
        pd.DataFrame: A DataFrame containing the merged data from messages and categories.

    Example:
        # Load and merge data
        data = load_data('messages.csv', 'categories.csv')
    """
    # Load the messages data from the CSV file
    messages = pd.read_csv(messages_filepath)

    # Load the categories data from the CSV file
    categories = pd.read_csv(categories_filepath)

    # Merge the messages and categories DataFrames on the 'id' column using an inner join
    df = pd.merge(messages, categories, on='id', how='inner')

    return df


def clean_data(df):
    """
    Clean and preprocess the input DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing raw data.

    Returns:
        pandas.DataFrame: A cleaned and preprocessed DataFrame.

    Example:
        # Clean the raw data
        cleaned_df = clean_data(raw_df)
    """
    # Split the 'categories' column into separate columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Extract column names from the first row
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    # Convert category values to binary (0 or 1)
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: int(x[-1]))
    
    # Drop the original 'categories' column
    df.drop(columns=['categories'], inplace=True)
    
    # Concatenate the DataFrames
    df = pd.concat([df, categories], axis=1)
    
    # Group and count the 'related' column
    df.groupby("related").count()
    
    # Replace '2' values in the 'related' column with '1'
    df["related"] = df["related"].apply(lambda x: 1 if x == 2 else x)
    
    # Remove columns with all NaN values
    df = df.dropna(axis=1, how='all')
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    return df



def save_data(df, database_filename):
    """
    Save a pandas DataFrame to a SQLite database.

    Args:
        df (pandas.DataFrame): The DataFrame to be saved to the database.
        database_filename (str): The file path of the SQLite database where the DataFrame will be stored.

    Returns:
        None

    Example:
        # Save a DataFrame to an SQLite database
        save_data(df, 'my_data.db')
    """
    # Create a connection to the SQLite database
    engine = create_engine('sqlite:///' + database_filename)

    # Save the DataFrame to the database, using the table name 'df' and excluding the index
    df.to_sql('DisasterResponse_table', engine, index=False,if_exists='replace')
    
    pass

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()