import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

import task_1
import task_2
import task_3
from data_handler import DataHandler
from preproccessing import Preproccessing

if __name__ == "__main__":
    # temp()

    train_path = './data_files/agoda_cancellation_train.csv'
    test1_path = './data_files/Agoda_Test_1.csv'
    test2_path = './data_files/Agoda_Test_2.csv'

    # Question 1 - Order Cancelation
    # task_1 = task_1.Task1()
    # task_1.run(train_path, test1_path)
    #
    # # Question 2 - Price
    # task_2 = task_2.Task2()
    # task_2.run(train_path, test2_path)

    # Question 3 + 4
    preproccessing = Preproccessing()
    data_handler = DataHandler()

    train_df = data_handler.load_data(train_path, True)
    train_df = preproccessing.preprocess_data(train_df, True)
    train_df.drop("h_booking_id", axis=1, inplace=True)
    train_df = train_df.drop_duplicates()

    task_3 = task_3.Task3()
    task_3.run(train_df)
