import pandas as pd
import numpy as np
import pyarrow as pa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, directed_hausdorff
from skimage import metrics

from tqdm import tqdm


class generate_var:
    def __init__(self, df, real, estimate, id_var, target_var):
        
        self.df = df
        self.real = real
        self.estimate = estimate
        self.id_var = id_var
        self.target_var = target_var


    def var_dtw(self):
        df = pa.Table.from_pandas(self.df)
        estimated_routes = df.column(self.estimate).to_pylist()
        real_routes = df.column(self.real).to_pylist()

        # Calcular la distancia DTW para cada par de rutas
        dtw = []
        for est_route, real_route in zip(estimated_routes, real_routes):
            distance, _ = fastdtw(est_route, real_route, dist=euclidean)
            dtw.append(distance)

        
        return dtw

    def get_lcs_length(self):
        
        # Code adaapted from Geek4Geek website
        # https://www.geeksforgeeks.org/longest-common-subsequence-dp-4/

        lcs = []
        for idx, row in tqdm(self.df[[self.estimate, self.real]].iterrows()):
            m = len(row[self.estimate])
            n = len(row[self.real])

            # Initializing a matrix of size (m+1)*(n+1)
            dp = [[0] * (n + 1) for x in range(m + 1)]

            # Building dp[m+1][n+1] in bottom-up fashion
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if (row[self.estimate][i - 1] == row[self.real][j - 1]).all():
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j],
                                    dp[i][j - 1])

            # dp[m][n] contains length of LCS for S1[0..m-1]
            # and S2[0..n-1]
            lcs.append(dp[m][n])
        return lcs




    # Python program for the above approach
    def levenshtein_two_matrix_rows(self):
        
        # Code adaapted from Geek4Geek website
        # https://www.geeksforgeeks.org/introduction-to-levenshtein-distance/
        
        # Get the lengths of the input strings
        leven = []
        for idx, row in tqdm(self.df[[self.estimate, self.real]].iterrows()):

            m = len(row[self.estimate])
            n = len(row[self.real])

            # Initialize two rows for dynamic programming
            prev_row = [j for j in range(n + 1)]
            curr_row = [0] * (n + 1)

            # Dynamic programming to fill the matrix
            for i in range(1, m + 1):
                # Initialize the first element of the current row
                curr_row[0] = i

                for j in range(1, n + 1):
                    if (row[self.estimate][i - 1] == row[self.real][j - 1]).all():
                        # Characters match, no operation needed
                        curr_row[j] = prev_row[j - 1]
                    else:
                        # Choose the minimum cost operation
                        curr_row[j] = 1 + min(
                            curr_row[j - 1],  # Insert
                            prev_row[j],      # Remove
                            prev_row[j - 1]    # Replace
                        )

                # Update the previous row with the current row
                prev_row = curr_row.copy()

            # The final element in the last row contains the Levenshtein distance
            leven.append(curr_row[n])
            
        return leven
    
    

    def var_hausdorff(self):
            
        # Code adaapted from Geek4Geek website
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html
        haus = []
        for idx, row in tqdm(self.df[[self.estimate, self.real]].iterrows()):
            estimate = np.vstack(np.array(row[self.estimate]))
            real = np.vstack(np.array(row[self.real]))
            dist = directed_hausdorff(estimate, real)[0]
            haus.append(dist)
        return haus
    
    def get_pandas_dataframe(self):
        
        print('hausdorff')
        hausdorff = self.var_hausdorff()
        print('levenshtein')
        levenshtein = self.levenshtein_two_matrix_rows()
        print('lcs')
        lcs = self.get_lcs_length()
        print('dtw')
        dtw = self.var_dtw()

        data = pd.DataFrame({
            'id':self.df[self.id_var],
            'target':self.df[self.target_var],
            'levenshtein':levenshtein,
            'lcs':lcs,
            'dtw':dtw,
            'hausdorff':hausdorff
        })

        return data
