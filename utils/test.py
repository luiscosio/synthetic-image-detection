import pandas
ima = pandas.read_parquet('../data/dalle-3-dataset/data/train-00000-of-00046-0b86771d32a801d8.parquet')
print(ima.shape)
print(ima.head(2))
