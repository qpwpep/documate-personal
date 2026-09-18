import numpy as np
pixels = np.arange(24).reshape(2, 4, 3)
gains = np.array([1.0, 0.5, 2.0])
adjusted = pixels * gains
scores = np.array([[0.2, 0.8, 0.8], [0.9, 0.1, 0.0]])
winners = scores.argmax(axis=1)
flat_batches = pixels.reshape(2, -1)
