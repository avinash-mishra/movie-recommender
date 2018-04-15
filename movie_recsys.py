import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM


# fetch data and format it
data = fetch_movielens(min_rating=4.0)

# print data
print(data)
print(repr(data['train']))
print(repr(data['test']))

# use 3 different loss functions (so 3 different models), compare results, print results for the best one. - Available loss functions are warp, logistic , bpr and warp-kos

# create model
model = LightFM(loss='warp')
# train model
model.fit(data['train'], epochs=30, num_threads=2)