import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


def create_tables(df):
    X = df.drop(columns=['order_canceled'])
    y = df['order_canceled']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)

    importances = clf.feature_importances_

    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
    plt.title('Feature Importance')
    plt.show()

    # corr
    corr = train_df.corr()['cancellation']
    # Sort by absolute values
    sorted_corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
    print(sorted_corr)
