# Email classifier results: laptop-8b, rules only

Test emails: 33. Accuracy: **93.9%**. Unusable replies: 0.

| Class | Precision | Recall | Emails |
|-------|-----------|--------|--------|
| Priority | 84.6% | 100.0% | 11 |
| Updates | 100.0% | 100.0% | 11 |
| Promotions | 100.0% | 81.8% | 11 |

| Difficulty | Accuracy |
|------------|----------|
| easy | 100.0% |
| hard | 66.7% |

Confusion matrix (rows = true label, columns = predicted):

```
predicted   Priority  Updates  Promotions
true                                     
Priority          11        0           0
Updates            0       11           0
Promotions         2        0           9
```
