# Email classifier results: laptop-16gb, rules only

Test emails: 33. Accuracy: **100.0%**. Unusable replies: 0.

| Class | Precision | Recall | Emails |
|-------|-----------|--------|--------|
| Priority | 100.0% | 100.0% | 11 |
| Updates | 100.0% | 100.0% | 11 |
| Promotions | 100.0% | 100.0% | 11 |

| Difficulty | Accuracy |
|------------|----------|
| easy | 100.0% |
| hard | 100.0% |

Confusion matrix (rows = true label, columns = predicted):

```
predicted   Priority  Updates  Promotions
true                                     
Priority          11        0           0
Updates            0       11           0
Promotions         0        0          11
```
