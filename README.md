# Predict-Future-Sales

kaggle隊伍名稱：n96071180  
kaggle kernal連結：https://www.kaggle.com/n96071180/predict-future-sales  

# data preprocessing
(1) 把outlier刪除，只取  
item_price > 0 & item_price < 100000  
item_cnt_day < 800 & item_cnt_day >= 0  
(2) convert the raw sales data to monthly sales  
(3) Merge the monthly sales data to the test data, Remove the categorical data from our test data  
(4) create the actual training set, and create the actual test set  

# model selection
(1) Create the model using the LSTM  
(2) Create the model using the XGB  
(3) Create the model using the GradientBoostingRegressor  

# test prediciton
Get the test set predictions and clip values to the specified range  



# 比較
使用了3個方法去實作  
GradientBoostingRegressor這個方法的準確值最低，score = 1.48  
XGB這個方法準確值尚可，score = 1.035，執行速度快  
LSTM這個方法準確值佳，score = 1.025，執行速度比XGB還要慢  

