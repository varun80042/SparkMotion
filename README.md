# SparkMotion

### Set up a TCP server on port 9999 to ingest data, using the following command 
```
ncat -l 9999
```

### Set up a Spark Streaming job, using the following command 
```
spark-submit --conf "spark.pyspark.python={PYTHON_EXE_PATH}" spark-solution.py
```

Run the command ```python -m tools.main``` for a model checkpoint. 
