# Titanic survivors: basic Python approach
Function-based approach to perform the data preparation tasks.  All functions are stored under `functions.py` and each function is limited to a single task.  While `functions.py` could be run as a main method, the main method is currently a Jupyter notebook for interactiveness.  This could change in the future.

The following data preparation tasks are considered
- [ ] Label encoding of non-numerical values:  v1 with hard-coded logic and no `LabelEncoder`

        - [X] Sex
        - [X] Embarked
        - [ ] Cabin
        
- [ ] Imputation of missing values: v1 with `KNNImputer`

        - [X] Age
        - [ ] Cabin
        - [ ] Embarked

- [ ] Creation of new features

        - [X] Agegroup
        - [X] Familysize
        - [X] haschildren: contains some edgecases that are not correctly covered
        - [X] hasparents: contains some edgecases that are not correctly covered
        - [X] hasspouse: contains some edgecases that are not correctly covered
        - [X] issinglechild