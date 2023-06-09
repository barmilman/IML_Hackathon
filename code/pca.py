import numpy as np
from matplotlib import pyplot as plt


def run_pca(data):
    # Assuming you have the data in a NumPy array called 'data'
    covariance_matrix = np.cov(data.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    n_components = np.count_nonzero(eigenvalues)

    # Sort the eigenvalues in descending order
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    # Plot the eigenvalues
    fig = plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues_sorted, marker='o', linestyle='-', color='b')
    plt.xlabel('Component')
    plt.xlim(0, 20)
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalues of Covariance Matrix')
    plt.grid(True)
    fig.show()
    fig.savefig('Eigenvalues of Covariance Matrix.png')

    from sklearn.decomposition import PCA
    import pandas as pd

    n_components = 4
    # Instantiate the PCA object with the desired number of components
    pca = PCA(n_components=n_components)

    # Fit the data to the PCA model
    pca.fit(data)

    # Transform the data to the lower-dimensional space
    data_transformed = pca.transform(data)

    # Create a DataFrame to store the transformed data
    column_names = ['PC{}'.format(i + 1) for i in range(n_components)]
    df_transformed = pd.DataFrame(data_transformed, columns=column_names)

    # The resulting DataFrame 'df_transformed' contains the transformed data
    # with the specified number of components (in this case, 2 components)

    explained_variance_ratio = pca.explained_variance_ratio_
    print(explained_variance_ratio)
    return df_transformed
