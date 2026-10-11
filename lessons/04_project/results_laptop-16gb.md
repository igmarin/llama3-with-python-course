# Email classifier results: laptop-16gb

Test emails: 33. Accuracy: **90.9%**. Unusable replies: 0.

| Class | Precision | Recall | Emails |
|-------|-----------|--------|--------|
| Priority | 83.3% | 90.9% | 11 |
| Updates | 91.7% | 100.0% | 11 |
| Promotions | 100.0% | 81.8% | 11 |

| Difficulty | Accuracy |
|------------|----------|
| easy | 100.0% |
| hard | 50.0% |

Confusion matrix (rows = true label, columns = predicted):

```
predicted   Priority  Updates  Promotions
true                                     
Priority          10        1           0
Updates            0       11           0
Promotions         2        0           9
```
