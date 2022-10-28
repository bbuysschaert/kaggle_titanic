# Author: Bram Buysschaert  

# Dependencies
import pandas as pd
import numpy as np

from typing import List, Tuple

#### Low-level functions
########################

def clean_titanic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Central function that cleans the Titanic survivors data.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with the Titanic survivor data
    
    Returns
    -------
    df : pd.DataFrame
        Clean dataframe
    """
    # Currently no cleaning steps defined

    return df

def basicimpute_titanic(df: pd.DataFrame, cols_encode: List[str] = []) -> Tuple[pd.DataFrame, dict]:
    """
    Central function that performs the imputation for any missing data in the Titanic survivors dataset.
    v1:  wrapper around `sklearn.impute.KNNImputer`

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with the Titanic survivor data
    cols_encode : list(str), optional
        List with the column names that need to be encoded before the imputation process
    
    Returns
    -------
    df : pd.DataFrame
        Titanic survivor data with any missing data imputed
    encoders : dict
        Dictionnary with the LabelEncoder objects per column on which they were used
    """
    from sklearn.impute import KNNImputer

    # Unsure yet if warranted.  Will cause issues for larger datasets
    df = df.copy(deep=True)
    assert df.index.name == 'PassengerId'

    # Encoding
    ##########
    df['Sex'] = [1 if val == 'male' else 0 if val == 'female' else np.nan for val in df['Sex']]
    df['Embarked'] = [0 if val == 'C' else 1 if val == 'S' else 2 if val == 'Q' else np.nan for val in df['Embarked']]
    
    # Imputation
    ############
    # This has a fixed sets of columns
    cols_impute = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    # Note: Cabin is not imputed at the moment!!!
    # Note: PassengerID is assumed to be the index of the dataframe
    cols_other = [cc for cc in df.columns if cc not in cols_impute]
    
    imputer = KNNImputer(n_neighbors=5,
                        weights='distance',
                        add_indicator=False
                        )
    # Apply imputer
    temp = pd.DataFrame(imputer.fit_transform(df.reset_index().loc[:, ['PassengerId'] + cols_impute]),
                         columns = ['PassengerId'] + cols_impute
                        )
    # Correct the datatypes after imputation
    for cc in ['PassengerId'] + cols_impute:
        if cc != 'Fare':
            temp[cc] = temp[cc].astype('int')
        
    # Reconstruct the index
    temp = temp.set_index('PassengerId', drop=True)
    
    # Join with original dataframe to retain missing columns
    df = (df.loc[:, cols_other]
            .join(temp,
                  on='PassengerId',
                  how= 'left'
                  )
         )
    return df

def impute_titanic(df: pd.DataFrame, cols_encode: List[str] = []) -> Tuple[pd.DataFrame, dict]:
    """
    Central function that performs the imputation for any missing data in the Titanic survivors dataset.
    v1:  wrapper around `sklearn.impute.KNNImputer`

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with the Titanic survivor data
    cols_encode : list(str), optional
        List with the column names that need to be encoded before the imputation process
    
    Returns
    -------
    df : pd.DataFrame
        Titanic survivor data with any missing data imputed
    encoders : dict
        Dictionnary with the LabelEncoder objects per column on which they were used
    """
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import LabelEncoder

    # Encoding
    ##########
    encoders = {}
    if len(cols_encode) > 0:
        for cc in cols_encode:
            encoder = LabelEncoder()
            df[cc] = encoder.fit_transform(df[cc].astype(str))
            encoders[cc] = encoder
    else:
        print('No encoding specified')
    
    # Imputation
    ############
    # This has a fixed sets of columns
    cols_impute = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    # Note: Cabin is not imputed at the moment!!!
    # Note: PassengerID is included for a join with the original dataframe df later
    cols_other = [cc for cc in df.columns if cc not in cols_impute]
    
    imputer = KNNImputer(n_neighbors=3,
                        weights='uniform',
                        add_indicator=True
                        )
    # Apply imputer    
    temp = pd.DataFrame(imputer.fit_transform(df.loc[:, cols_impute]),
                        columns = cols_impute + ['Valueimputed']
                        )
    # Join with original dataframe to retain missing columns
    df = (df.loc[:, ['PassengerId'] + cols_other]
            .join(temp,
                  on='PassengerId',
                  how= 'left'
                  )
         )
    return df, encoders


def addfeature_familysize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine the family size of the person travelling feature.
    Equation is: df['SibSp'] + df['Parch'] + 1

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with the Titanic survivor data
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with added feature
    """
    df['Familysize'] = df['SibSp'] + df['Parch'] + 1
    return df

def addfeature_agegroup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine the age group of the person travelling
    Three categories are proposed:
    - child: if age < 18
    - elder: if age > 65
    - adult: else

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with the Titanic survivor data
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with added feature
    """
    df['Agegroup'] = ['child' if vv < 18 else 'elder' if vv > 65 else 'adult' for vv in df['Age']]
    return df

def addfeature_haschildren(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine whether the person travelling has children
    WARNING: Some edge cases are not covered, such as adults travelling with adult parents
    DEPENDENCY: Requires the feature "Agegroup"

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with the Titanic survivor data
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with added feature
    """
    df['haschildren'] = df.apply(lambda row : (row['Agegroup'] == 'adult') & (row['Parch'] > 0), axis=1)
    return df

def addfeature_hasparents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine whether the person travelling has parents
    WARNING: Does not correctly resolve edge cases when Parch > 2
    DEPENDENCY: Requires the feature "Agegroup"

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with the Titanic survivor data
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with added feature
    """
    df['hasparents'] = df.apply(lambda row : (row['Agegroup'] == 'child') & (row['Parch'] > 0), axis=1)
    return df

def addfeature_hasspouse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine whether the person travelling has a spouse
    WARNING: Does not correctly resolve edge cases when SibSp > 2
    DEPENDENCY: Requires the feature "Agegroup"

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with the Titanic survivor data
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with added feature
    """
    df['hasspouse'] = df.apply(lambda row : (row['Agegroup'] != 'child') & (row['SibSp'] > 0), axis=1)
    return df

def addfeature_issinglechild(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine whether the person travelling is a single child
    DEPENDENCY: Requires the feature "Agegroup"

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with the Titanic survivor data
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with added feature
    """
    df['issinglechild'] = df.apply(lambda row : (row['Agegroup'] == 'child') & (row['SibSp'] == 0), axis=1)
    return df

def encode_titanic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode all "object" columns with a label encoder

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with the Titanic survivor data
    
    Returns
    -------
    df : pd.DataFrame
        Titanic survivor data with all object columns label encoded
    encoders : dict
        Dictionnary with the LabelEncoder objects per column on which they were used
    """
    from sklearn.preprocessing import LabelEncoder

    # Store the datatypes of all columns as a dict
    dtypes = df.dtypes.to_dict().items()

    encoders = {}
    
    for cc, tt in dtypes:
        if tt == 'object':
            encoder = LabelEncoder()
            df[cc] = encoder.fit_transform(df[cc].astype(str))
            encoders[cc] = encoder
    return df, encoders

def dataprep_titanic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main method with all the preparation steps for the Titanic survivor dataset
    """
    # Clean and impute
    ##################
    df = clean_titanic(df)
    df = basicimpute_titanic(df, cols_encode=['Sex', 'Embarked'])

    # Add features
    ##############
    df = addfeature_familysize(df)
    df = addfeature_agegroup(df)

    df = addfeature_haschildren(df)
    df = addfeature_hasparents(df)
    df = addfeature_hasspouse(df)

    df = addfeature_issinglechild(df)

    # Encode string columns
    #######################
    # Ignore encoding for the moment

    #df, encoders2 = encode_titanic(df)
    #encoders = encoders1.update(encoders2)
    for cc, tt in df.dtypes.to_dict().items():
        if tt == 'bool':
            df[cc] = df[cc].astype('int')
        elif cc == 'Agegroup':
            df[cc] = [0 if val == 'child' else 1 if val == 'adult' else 2 if val == 'elder' else np.nan for val in df['Agegroup']]
    
    encoders = {}

    return df, encoders



if __name__ == '__main__':
    # Variables with pointers to the data
    file_train = './Data/test.csv'
    file_train_clean = ''
    file_test = './Data/train.csv'
    file_pred = ''

    # Extract
    df_train = pd.read_csv(file_train, sep=',', header='infer', index_col='PassengerId')
    df_test = pd.read_csv(file_test, sep=',', header='infer', index_col='PassengerId')

    # Transform
    df_train, enc_train = dataprep_titanic(df_train)
    df_test, enc_test = dataprep_titanic(df_test)

    # Load