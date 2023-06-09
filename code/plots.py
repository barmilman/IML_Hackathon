import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def create(train_df):
    # Percentage of cancellations by days booked before

    percentage_df = train_df.groupby('booked_days_before')['order_cancelled'].agg(['sum', 'count']).reset_index()
    percentage_df['percentage'] = (percentage_df['sum'] / percentage_df['count']) * 100

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=percentage_df, x='booked_days_before', y='percentage')
    plt.title('Percentage of cancellations by days booked before')
    plt.xlabel('Booked Days Before')
    plt.ylabel('Cancellation Percentage')
    plt.show()

    # Percentage of Cancellations by Duration of Stay

    grouped_df = train_df.groupby('duration_days')['order_cancelled'].agg(['sum', 'count']).reset_index()

    # Rename the columns for clarity
    grouped_df.columns = ['duration_days', 'num_cancellations', 'total_bookings']

    grouped_df['percentage_cancellations'] = (grouped_df['num_cancellations'] / grouped_df['total_bookings']) * 100

    plt.figure(figsize=(10, 6))
    plt.plot(grouped_df['duration_days'], grouped_df['percentage_cancellations'], marker='o')
    plt.xlabel('Duration of Stay (days)')
    plt.ylabel('Percentage of Cancellations')
    plt.title('Percentage of Cancellations by Duration of Stay')
    plt.show()

    X = train_df.drop(columns=['order_cancelled'])
    y = train_df['order_cancelled']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)

    importances = clf.feature_importances_

    # df for visualization
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(5, 5))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

    # corr
    corr = train_df.corr()['order_cancelled']
    sorted_corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
    print(sorted_corr)
