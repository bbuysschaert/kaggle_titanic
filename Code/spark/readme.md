# Titanic survivors: PySpark approach

Perform the data preparation steps with a PySpark dataframe.  Most of the logic of the other approaches need to be translated to use `pyspark.sql.functions` functionality instead of Pandas functionality.  

Initially, the goal is to only have the data prepation steps done with PySpark.  The ML training and inference could then still be done on a Pandas dataframe (converted from the PySpark dataframe).  In a section attempt, the data preparation could be converted to a Spark ML pipeline, including Spark ML for training and inference.  This is still to be decided...

While the `environment.yml` contains additional modules for the Spark setup, it is not contained (yet?) within the configuration file.  It has strong dependencies to Java, Hadoop file configs, and the Spark tarball itself.  As such, it was installed manually and separatelly.  [A good tutorial for Installing it on a Windows 11 machine is found here](https://linuxhint.com/install-apache-spark-windows-11/).

_Disclaimer_: I am aware that PySpark is overkill for this exercise, but this setup should, hopefully, demonstrate what PySpark could do.  More precisely, help my understand minor differences when working with Spark ML.

