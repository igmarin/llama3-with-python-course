# Message classifier results: laptop-16gb, rules only

Challenge messages: 60. Accuracy: **90.0%**. Unusable replies: 0.

| Class | Precision | Recall | Messages |
|-------|-----------|--------|--------|
| Repair | 100.0% | 95.0% | 20 |
| Order | 93.8% | 75.0% | 20 |
| Feedback | 80.0% | 100.0% | 20 |

| Difficulty | Accuracy |
|------------|----------|
| hard | 90.0% |

Confusion matrix (rows = true label, columns = predicted):

```
predicted  Repair  Order  Feedback
true                              
Repair         19      1         0
Order           0     15         5
Feedback        0      0        20
```
