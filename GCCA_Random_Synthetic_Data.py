#Title: GCCA_Random_Synthetic_Data.py
import numpy as np
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt

# generate simulated data (similar to the data in the paper)
np.random.seed(42)
N = 2000  # sample sizea
d1, d2, d3 = 50, 50, 50  # two views of features (d1, d2, d3)

# generate related data
Z = np.random.randn(N, 50)  # shared latent variables
W =  np.random.randn(N, d1) + 0.1 * np.random.randn(N, d1) # Random language 1 embedding
np.random.seed(22)
X =  np.random.randn(N, d2) + 0.1 * np.random.randn(N, d2) # Random language 2 embedding 
Y = Z @ np.random.randn(50, d3) + 0.1 * np.random.randn(N, d3) # Random language 3 embedding 



# train CCA
cca = CCA(n_components=50, max_iter=2000)
W_c, X_c = cca.fit_transform(W, X)  # project to the CCA-related space

# calculate and project the correlation between the projected data
corrs = [np.corrcoef(W_c[:, i], X_c[:, i])[0, 1] for i in range(50)]

# result visualization
plt.figure(figsize=(8, 4))
plt.bar(range(1, 51), corrs)
plt.xlabel("GCCA Component")
plt.ylabel("correlation coefficient")
plt.title("GCCA projection correlation")
plt.show()

# display the first 3 correlations
print("first 3 correlations of the GCCA projection: ", corrs[:3])
