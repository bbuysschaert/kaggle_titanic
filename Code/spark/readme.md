# Titanic survivors: PySpark approach

Perform the data preparation steps with a PySpark dataframe.  Most of the logic of the other approaches need to be translated to use `pyspark.sql.functions` functionality instead of Pandas functionality.  

Initially, the goal is to only have the data prepation steps done with PySpark.  The ML training and inference could then still be done on a Pandas dataframe (converted from the PySpark dataframe).  In a section attempt, the data preparation could be converted to a Spark ML pipeline, including Spark ML for training and inference.  This is still to be decided...

_Disclaimer_: I am aware that PySpark is overkill for this exercise, but this setup should, hopefully, demonstrate what PySpark could do.  More precisely, help my understand minor differences when working with Spark ML.

