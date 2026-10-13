# Message classifier results: laptop-16gb, rules plus six worked examples

Test messages: 33. Accuracy: **100.0%**. Unusable replies: 0.

| Class | Precision | Recall | Messages |
|-------|-----------|--------|--------|
| Repair | 100.0% | 100.0% | 11 |
| Order | 100.0% | 100.0% | 11 |
| Feedback | 100.0% | 100.0% | 11 |

| Difficulty | Accuracy |
|------------|----------|
| easy | 100.0% |
| hard | 100.0% |

Confusion matrix (rows = true label, columns = predicted):

```
predicted  Repair  Order  Feedback
true                              
Repair         11      0         0
Order           0     11         0
Feedback        0      0        11
```
