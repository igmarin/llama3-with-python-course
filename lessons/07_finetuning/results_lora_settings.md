# LoRA settings compared: laptop-16gb

| Setting | Trainable weights | Final validation loss | Challenge accuracy (60) |
|---------|-------------------|-----------------------|-------------------------|
| Lesson 7.2 default: rank 8, scale 20, all layers | 6.947M | 0.001 | 93.3% |
| Rank 12, alpha 8 (scale 0.67), q/v only | 1.966M | 0.029 | 85.0% |
| Rank 2, alpha 4 (scale 2), q/v only, 250 tokens | 0.328M | 0.040 | 83.3% |
| Rank 2, scale 20, q/v only, 250 tokens | 0.328M | 0.010 | 95.0% |
