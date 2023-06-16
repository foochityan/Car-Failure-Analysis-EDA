Fullname
Email

File Structure
root
 - eda.ipynb
-- readme.md

The pipeline consists of 1 class Config. To configure the pipeline parameters, edit config.py with the desired settings before running. The pipeline is ran by executing run.sh.

Configurations Available

- scale - Scales the data based on training set
- transform - Log transformation on Fuel consumption
- balance - Balances the label proportions
- cv - Number of cross validation folds to perform
- perform_cv - Whether or not to perform cross validation
- return_top_models - integer, specifies how many models to return for tuning. If None, no models will be tuned.
- random_tune - If True, uses Randomize Search CV. Else, uses Grid Search CV
- models - The recommended models for the dataset
- base_models - Baseline models to examine how well different types of classifiers perform on the data
- model_params - Parameter grid for hyperparameter tuning

### Key findings of EDA

1. Negative values present in RPM
2. Fuel consumption is slightly right skewed
3. Temperature can be label encoded into high and low temperature because of its distribution
4. Large number of unique models, should not be one hot encoded
5. Varied distributions of categorical values, tree based algorithms should perform well on this data
6. Failures can be merged into 1 column, as 1 car can only have a single type of failure
7. Imbalance of target labels (>80% of cars have no failures), down-sampling of data should be performed, but at the expense of losing accuracy

The pipeline consists of 6 steps

1. Cleaning
2. Encoding
3. Preparing
4. Model Training
5. Model Tuning
6. Evaluation

We begin the pipeline process with 10,081 rows and 14 columns of data

### Cleaning of Data

From the EDA conducted, the following errors were corrected.
| Column | Error Description | Correction Made |
| ------ | ------ | ------- |
| Membership | Contains NA values | Replaced with "None" |
| Temperature | Stored as a string | Extracted the temperature values encoded to High Temp and Low Temp|
| Car ID | Unused column | Dropped from the data |
| RPM | Negative Values | Dropped rows with negative RPM |

For Temperature, the histogram of its values showed a clean boundary between 2 groups of values. Within each group, its variance is low. Hence, encoding it into "High" and "Low" temperature would reduce noise in the features.

![[Pasted image 20230115175519.png]]
After cleaning, there are 9,857 rows and 13 columns.

### Encoding

After encoding of Temperature, only RPM and Fuel consumption remained as numerical features. The rest of the features needs to be encoded. To ensure low dimensionality after encoding, features with a large number of unique values were encoded using mean weight encoding, while one hot encoding was used for the rest of the features.

Additionally, we merged the 5 Fault columns into one, as we found in the EDA that a car can only have 1 type of fault. Doing this reduces the number of classifiers that we have to train.

After encoding, The data has 9,857 rows and 23 columns.

The summary of feature processing is as follows
| Column | Process |
| ------ | ------ |
| Car ID | Dropped |
| Model | Mean weight encoded |
| Temperature | Value extracted, then label encoded |
| RPM | Dropped negative values, with option to standardise and log transform |
| Factory | One hot encoded |
| Usage | One hot encoded |
| Fuel consumption | Option to standardise|
| Membership | NA values replace with "None", then one hot encoded |
| Failure Columns | Joined into 1 Faults column |

### Preparing

3 options are available for preparing of data.

1. Log transformation of fuel consumption, as it is slightly right skewed
2. Balancing of target labels
3. Standardisation of data

After the desired operations are carried out, the data is split in train and test sets for model training.

Before splitting into training and testing sets,

- If the data is balanced, there are a total of 1,496 rows
- If not, there are a total of 9,857 rows.

After splitting,

- If the data is balanced, there are a total of 1,047 rows in the training set, and 449 rows in the test set (70% Train, 30% Test)
- If not, there are a total of 6,899 rows in the training set, and 2,958 rows in the test set. (70% Train, 30% Test)

.
