## Why Human-level Performance

![performance](https://raw.githubusercontent.com/seanliu96/deeplearning.ai/master/COURSE%203%20Structuring%20Machine%20Learning%20Projects/week1/Comparing%20To%20Human-level%20Performance/images/performance.png)

Machine learning progresses slowly when it surpasses human-level performance. 

One of the reason is that human-level performance can be close to Bayes optimal error, especially for natural perception problem (Bayes optimal error is defined as the best possible error. In other words, it means that any functions mapping from x to y can’t surpass a certain level of accuracy.)

Also, when the performance of machine learning is worse than the performance of humans, you can improve it with different tools. They are harder to use once its surpasses human-level performance. These tools are

- Get labeled data from humans
- Gain insight from manual error analysis: Why did a person get this right
- Better analysis of bias / variance

## Avoidable bias

If you want to improve the performance of the training set but you can’t do better than the Bayes error otherwise the training set is overfitting. By knowing the Bayes error, it is easier to focus on whether bias or variance avoidance tactics will improve the performance of the model.

### Scenario A

There is a 7% gap between the performance of the training set and the human level error. It means that the algorithm isn’t fitting well with the training set since the target is around 1%. To resolve the issue, we use bias reduction technique such as training a bigger neural network or running the training set longer.

### Scenario B

The training set is doing good since there is only a 0.5% difference with the human level error. The difference between the training set and the human level error is called avoidable bias. The focus here is to reduce the variance since the difference between the training error and the development error is 2%. To resolve the issue, we use variance reduction technique such as regularization or have a bigger training set.

## Understanding Human-level Performance

human error(proxy for Bayes error) <—(avoidable bias)—> training error <—(variance)—> dev error

### Scenario A

|                                     | Classification Error (%) |
| ----------------------------------- | ------------------------ |
| human error1(proxy for Bayes error) | 0.5                      |
| human error2                        | 0.7                      |
| human error3                        | 1                        |
| training error                      | 5                        |
| development error                   | 6                        |

The choice of human-level performance doesn’t have an impact. The avoidable bias is between 4%-4.5% and the variance is 1%. Therefore, the focus should be on **bias reduction** technique.

### Scenario B

|                                     | Classification Error (%) |
| ----------------------------------- | ------------------------ |
| human error1(proxy for Bayes error) | 0.5                      |
| human error2                        | 0.7                      |
| human error3                        | 1                        |
| training error                      | 1                        |
| development error                   | 5                        |

The choice of human-level performance doesn’t have an impact. The avoidable bias is between 0%-0.5% and the variance is 4%. Therefore, the focus should be on **variance reduction** technique.

### Scenario C

|                                     | Classification Error (%) |
| ----------------------------------- | ------------------------ |
| human error1(proxy for Bayes error) | 0.5                      |
| training error                      | 0.7                      |
| development error                   | 0.8                      |

The estimate for Bayes error has to be 0.5% since you can’t go lower than the human-level performance otherwise the training set is overfitting. Also, the avoidable bias is 0.2% and the variance is 0.1%. Therefore, the focus should be on **bias reduction** technique

### Summary of Bias/Variance with Human-level Performance

- Human-level error – proxy for Bayes error
- If the difference between human-level error and the training error is bigger than the difference between the training error and the development error. The focus should be on bias reduction technique
- If the difference between training error and the development error is bigger than the difference between the human-level error and the training error. The focus should be on variance reduction technique

## Surpassing Human-level Performance

### Scenario A

|                   | Classification Error(%) |
| ----------------- | ----------------------- |
| team of humans    | 0.5                     |
| one human         | 1.0                     |
| traning error     | 0.6                     |
| development error | 0.8                     |

The Bayes error is 0.5% (surpassing 1.0%), therefore the available bias is 0.1% et the variance is 0.2%.

### Scenario B

|                   | Classification Error(%) |
| ----------------- | ----------------------- |
| team of humans    | 0.5                     |
| one human         | 1.0                     |
| traning error     | 0.3                     |
| development error | 0.4                     |

There is not enough information to know if bias reduction or variance reduction has to be done on the algorithm. It doesn’t mean that the model cannot be improve, it means that the conventional ways to know if bias reduction or variance reduction are not working in this case.

### Problems Where ML Significantly Surpasses Human-level Performance, especially with structured data

- Online advertising
- Product recommendations
- Logistrics(predicting transit time)
- Loan approvals

## Improving Your Model Performance

### Two fundamental Assumptings of Supervised Learning

- You can fit the traning set pretty well (avoidable bias is low)
  - train bigger model
  - train longer, better optimization algorithms
  - neural networks architecture/hyperparameters search
- The training set performance generalizes pretty well to the dev/test set (variance is low)
  - more data
  - regularization
  - neural networks architecture/hyperparameters search

