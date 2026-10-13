# Message classifier results: laptop-8b, rules plus six worked examples

Challenge messages: 60. Accuracy: **98.3%**. Unusable replies: 0.

| Class | Precision | Recall | Messages |
|-------|-----------|--------|--------|
| Repair | 100.0% | 100.0% | 20 |
| Order | 100.0% | 95.0% | 20 |
| Feedback | 95.2% | 100.0% | 20 |

| Difficulty | Accuracy |
|------------|----------|
| hard | 98.3% |

Confusion matrix (rows = true label, columns = predicted):

```
predicted  Repair  Order  Feedback
true                              
Repair         20      0         0
Order           0     19         1
Feedback        0      0        20
```
