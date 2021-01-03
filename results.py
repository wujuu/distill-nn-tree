import numpy as np
import seaborn as sns
import pandas as pd


if __name__ == '__main__':

     x = [('model_1_x_64_3_x_1024_dense', 0.8986, 0.6369),
          ('model_1_x_64_7_x_512_dense', 0.9101, 0.7797),
          ('model_1_x_64_30_x_256_dense', 0.7124, 0.7261),
          ('model_1_x_128_2_x_1024_dense', 0.9178, 0.7952),
          ('model_1_x_128_5_x_512_dense', 0.9206, 0.7947),
          ('model_1_x_128_15_x_256_dense', 0.9071, 0.7795)]

     pd = pd.DataFrame(x, columns=['nn_architecture', 'nn_accuracy', 'distilled_tree_accuracy'])
     pd = pd.set_index('nn_architecture')
     pd = pd.sort_values('nn_accuracy')
     print(pd)


