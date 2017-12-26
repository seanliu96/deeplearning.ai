## Carrying Out Error Analysis

Error analysis from the dev set to evaluate a classifier on another class:

- get ~100 mislabeled dev set examples
- count up how many belong to another class

## Cleaning Up Incorrectly Labeled Data

DL algorithms are quite robust to random errors in the training set

### Collecting Incorrect Dev/Test Set Examples

- apply same process to your dev and test sets to make sure they continue to come from the same distribution
- consider examining examples your algorithm got right as well as ones it got wrong
- train and dev/test data may now come from slightly different distributions

### Build Your First System Quickly, Then Iterate

The guideline is

1. set up dev/test set and metric
   - set up a target
2. build initial system quickly
   - training set: fit the parameters
   - development set: tune the parameters
   - test set: assess the performance
3. use bias/variance analysis & error analysis to prioritize next steps

