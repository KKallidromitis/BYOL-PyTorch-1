from utils.kmeans.per_image import BatchwiseKMeans
import torch
kmeans = BatchwiseKMeans(16)
x = torch.rand(4,100,256)
ck,c = kmeans.fit_transform(x)