![pink headphones](https://api.time.com/wp-content/uploads/2018/04/listening-to-music-headphones.jpg?quality=85&w=1600)

## Project Motivation
This study uses data analysis and machine learning in Pyspark to explore the issue of predicting churn. The project's aim is to be able to predict which users of a music platform, "Sparkify", will churn (downgrade to a free tier, or cancel the service). This would then allow their ‘marketing department’ to target these users with discounts or incentives to make them stay, and potentially save the business revenue.

This project originates as part of the Udacity DataScience Nanodegree programme, but predicting churn is a real world issue.

## Installation
-  Python 3
-  Spark 2.0 
In versions of Spark 3.0 or greater there may be issues with date formatting using the Sparkify notebook. In this case use the below statement to return the date-formatting to legacy version.
`spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")`
### Libraries Used/Dependencies
```
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import IntegerType, TimestampType
from pyspark.sql.functions import isnan, when, count, col, concat_ws, avg
from pyspark.sql.functions import asc, desc, explode, lit, min, max, split
from pyspark.sql.functions import sum as Fsum, udf, countDistinct, date_format
from pyspark.sql.functions import when, to_date, lag, coalesce

import random
import re
import datetime
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
```

### Files Contents
- Sparkify.ipynb - Jupyter notebook containing the exploration of the sparkify mini dataframe, data pre-processing, feature building and model implementation.
- [mini_sparkify_event_data.json.zip](https://github.com/alciamoncrieffe/Sparkify_Udacity/blob/main/mini_sparkify_event_data.json.zip "mini_sparkify_event_data.json.zip") - Mini event dataset used in the Sparkify.ipynb notebook. 
- [medium-sparkify-event-data.json.zip](https://github.com/alciamoncrieffe/Sparkify_Udacity/blob/main/medium-sparkify-event-data.json.zip "medium-sparkify-event-data.json.zip") - Medium event dataset also provided in case a potential user is interested in training the model on a larger dataset.
- https://medium.com/@alciamoncrieffe/predict-churn-on-your-music-platform-fbd3331040e0 - Blog post where I discuss the study in detail.

## Results Summary
 As part of this study I trained 2 classification models, with cross validation and tuning of the hyperparameters. Of the two models tested, the tuned RandomForestClassifier performed the best with the best AUC scores producing less than 17% error on the 128MB mini dataset that was used. It had better precision, and achieved a higher level of accuracy faster, even before tuning, while LogisticRegression model required more tuning.
 
 The next step would be to deploy/train AWS or IBM cloud with the medium sparkify dataset to see if what impact a larger dataset would have to the model accuracy.

## Contributing

Feel free to create new branches or contribute. I have not had time/resources to deploy to a web service so would be interested in how the predictions of a model trained on a larger dataset would differ from the above results.

## Licensing

See package license.txt file.

> Written with [StackEdit](https://stackedit.io/).
