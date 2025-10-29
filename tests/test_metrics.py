import numpy as np
from src.train import lift_at_k

def test_lift_at_k_basic():
    y_true = np.array([0,1,0,1,0,1,0,0,1,0])
    y_prob = np.array([0.1,0.9,0.2,0.8,0.3,0.7,0.4,0.5,0.6,0.05])
    lift = lift_at_k(y_true, y_prob, k=0.1)
    assert lift >= 1.0
