import pandas as pd
import numpy as np
from multiprocessing import Pool

# Generate big dataset
df = pd.DataFrame({
    "user": np.random.randint(1, 100000, 2000000),
    "value": np.random.rand(2000000)
})

def process(chunk):
    return chunk.groupby("user").value.mean()

# Split into chunks
chunks = np.array_split(df, 4)

with Pool(4) as p:
    results = p.map(process, chunks)

final = pd.concat(results)

print(final.head())
