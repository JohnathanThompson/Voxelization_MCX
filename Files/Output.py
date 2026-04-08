import jdata as jd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = jd.load("C:\\Users\\John Thompson\\Desktop\\Programs\\GroupProj\\result_0501.jnii")

# Extract and convert
fluence = np.array(data["NIFTIData"])

# Remove extra dims (time, source)
fluence = np.squeeze(fluence)

print("Shape:", fluence.shape)  # should be (83, 53, 181)

# Pick middle slice along Z
z_mid = fluence.shape[0] // 2

slice_xy = fluence[:, :, z_mid]

plt.imshow(slice_xy, cmap='hot')
plt.title(f"Fluence (Z slice = {z_mid})")
plt.colorbar(label="Fluence")
plt.show()