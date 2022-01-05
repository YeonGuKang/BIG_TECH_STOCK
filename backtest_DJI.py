import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler , RobustScaler , StandardScaler	, MaxAbsScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM , GRU
from tensorflow.keras.models import load_model
import wandb
import pprint

factor = []
val_loss_list = []
test_loss_list = []

test_loss = 0

class My_LSTM_simple:
    # 생성자
    def __init__(self,node,feature,scaler_method,epochs, batch_size, window_size):
        self.node = node
        self.feature = feature
        self.scaler_method = scaler_method
        self.epochs = epochs
        self.batch_size = batch_size
        self.window_size = window_size

        df_price = pd.read_csv('./거시경제지표_kospi200_스케일링.csv', encoding='cp949')
        scaler = scaler_method
        scale_cols = df_price.columns

        df_scaled = scaler.fit_transform(df_price[scale_cols])

        df_scaled = pd.DataFrame(df_scaled)
        df_scaled.columns = scale_cols

        TEST_SIZE = 400

        train = df_scaled[:-TEST_SIZE]
        test = df_scaled[-TEST_SIZE:]

        feature_cols = self.feature
        label_cols = ['Close']

        self.train_feature = train[feature_cols]
        self.train_label = train[label_cols]

        self.test_feature = test[feature_cols]
        self.test_label = test[label_cols]

        self.train_feature, self.train_label = self.make_dataset(self.train_feature, self.train_label, self.window_size)
        self.test_feature, self.test_label = self.make_dataset(self.test_feature, self.test_label, self.window_size)
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.train_feature, self.train_label, test_size=0.2)



    def make_dataset(self,data, label, window_size):
        feature_list = []
        label_list = []
        for i in range(len(data) - window_size):
            feature_list.append(np.array(data.iloc[i:i+window_size]))
            label_list.append(np.array(label.iloc[i+window_size]))

        # print(feature_list)
        # print(label_list)
        return np.array(feature_list), np.array(label_list)
            
    def train(self):
        model = Sequential()
        
        model.add(LSTM(self.node, 
                    input_shape=(self.train_feature.shape[1], self.train_feature.shape[2]), 
                    activation='relu', 
                    return_sequences=False)
                )
        
        model.add(Dense(1))

        model.summary()

        model.compile(loss='mean_squared_error', optimizer='adam')
        # early_stop = EarlyStopping(monitor='val_loss', patience=5)
        # filename = os.path.join(model_path, 'tmp_checkpoint.h5')
        # checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        history = model.fit(self.x_train, self.y_train, 
                            epochs=self.epochs, 
                            batch_size=self.batch_size,
                            validation_data=(self.x_valid, self.y_valid))
                            # callbacks=[early_stop, checkpoint])


        pred = model.predict(self.test_feature)
        loss = model.evaluate(self.test_feature,self.test_label)
        test_loss = loss
        wandb.log({"test_loss": test_loss})


        print(loss)

        plt.figure(figsize=(12, 9))
        plt.plot(self.test_label, label='actual')
        plt.plot(pred, label='prediction')
        plt.legend()
        plt.savefig(f'LSTM_simple_{self.epochs}_{self.node}_{self.window_size}.png')
        # plt.show()
        model.save(f'LSTM_simple_{self.epochs}_{self.node}_{self.window_size}.h5')

        factor.append("LSTM_simple")
        val_loss_list.append(history.history['val_loss'][self.epochs-1])
        test_loss_list.append(loss)

        

            
class My_LSTM_deep:
        # 생성자
    def __init__(self,nodes,feature,scaler_method,epochs, batch_size,window_size):
        self.feature = feature
        self.scaler_method = scaler_method
        self.epochs = epochs
        self.batch_size = batch_size
        self.window_size = window_size
        self.nodes = nodes

        df_price = pd.read_csv('./거시경제지표_kospi200_스케일링.csv', encoding='cp949')
        scaler = scaler_method
        scale_cols = df_price.columns

        df_scaled = scaler.fit_transform(df_price[scale_cols])

        df_scaled = pd.DataFrame(df_scaled)
        df_scaled.columns = scale_cols

        TEST_SIZE = 400

        train = df_scaled[:-TEST_SIZE]
        test = df_scaled[-TEST_SIZE:]

        feature_cols = self.feature
        label_cols = ['Close']

        self.train_feature = train[feature_cols]
        self.train_label = train[label_cols]

        self.test_feature = test[feature_cols]
        self.test_label = test[label_cols]

        self.train_feature, self.train_label = self.make_dataset(self.train_feature, self.train_label, self.window_size)
        self.test_feature, self.test_label = self.make_dataset(self.test_feature, self.test_label, self.window_size)
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.train_feature, self.train_label, test_size=0.2)



    def make_dataset(self,data, label, window_size):
        feature_list = []
        label_list = []
        for i in range(len(data) - window_size):
            feature_list.append(np.array(data.iloc[i:i+window_size]))
            label_list.append(np.array(label.iloc[i+window_size]))

        # print(feature_list)
        # print(label_list)
        return np.array(feature_list), np.array(label_list)
            
    def train(self):
        regression_LSTM = Sequential()

        regression_LSTM.add(LSTM(units=self.nodes,activation="relu", return_sequences=True, input_shape = (self.train_feature.shape[1], self.train_feature.shape[2])))
        
        regression_LSTM.add(LSTM(units=self.nodes, activation="relu", return_sequences=True))

        regression_LSTM.add(LSTM(units=self.nodes, activation="relu", return_sequences=True))

        regression_LSTM.add(LSTM(units=self.nodes, activation="relu"))

        regression_LSTM.add(Dense(units = 1))

        regression_LSTM.compile(optimizer='adam', loss='mean_squared_error')
        LSTM_history = regression_LSTM.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, shuffle=False,validation_data=(self.x_valid, self.y_valid))

        pred = regression_LSTM.predict(self.test_feature)
        loss = regression_LSTM.evaluate(self.test_feature,self.test_label)
        test_loss = loss
        wandb.log({"test_loss": test_loss})

        print(loss)

        plt.figure(figsize=(12, 9))
        plt.plot(self.test_label, label='actual')
        plt.plot(pred, label='prediction')
        plt.legend()
        plt.savefig(f'LSTM_deep_{self.epochs}_{self.nodes}_{self.window_size}.png')
        # plt.show()

        regression_LSTM.save(f'LSTM_deep_{self.epochs}_{self.nodes}_{self.window_size}.h5')

        factor.append("LSTM_deep")
        val_loss_list.append(LSTM_history.history['val_loss'][self.epochs-1])
        test_loss_list.append(loss)



class My_GRU_simple:
    # 생성자
    def __init__(self,node,feature,scaler_method,epochs, batch_size, window_size):
        self.node = node
        self.feature = feature
        self.scaler_method = scaler_method
        self.epochs = epochs
        self.batch_size = batch_size
        self.window_size = window_size

        df_price = pd.read_csv('./거시경제지표_kospi200_스케일링.csv', encoding='cp949')
        scaler = scaler_method
        scale_cols = df_price.columns

        df_scaled = scaler.fit_transform(df_price[scale_cols])

        df_scaled = pd.DataFrame(df_scaled)
        df_scaled.columns = scale_cols

        TEST_SIZE = 400

        train = df_scaled[:-TEST_SIZE]
        test = df_scaled[-TEST_SIZE:]

        feature_cols = self.feature
        label_cols = ['Close']

        self.train_feature = train[feature_cols]
        self.train_label = train[label_cols]

        self.test_feature = test[feature_cols]
        self.test_label = test[label_cols]

        self.train_feature, self.train_label = self.make_dataset(self.train_feature, self.train_label, self.window_size)
        self.test_feature, self.test_label = self.make_dataset(self.test_feature, self.test_label, self.window_size)
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.train_feature, self.train_label, test_size=0.2)



    def make_dataset(self,data, label, window_size):
        feature_list = []
        label_list = []
        for i in range(len(data) - window_size):
            feature_list.append(np.array(data.iloc[i:i+window_size]))
            label_list.append(np.array(label.iloc[i+window_size]))

        # print(feature_list)
        # print(label_list)
        return np.array(feature_list), np.array(label_list)
            
    def train(self):
        model = Sequential()
        
        model.add(GRU(self.node, 
                    input_shape=(self.train_feature.shape[1], self.train_feature.shape[2]), 
                    activation='relu', 
                    return_sequences=False)
                )
        
        model.add(Dense(1))

        model.summary()

        model.compile(loss='mean_squared_error', optimizer='adam')
        # early_stop = EarlyStopping(monitor='val_loss', patience=5)
        # filename = os.path.join(model_path, 'tmp_checkpoint.h5')
        # checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        history = model.fit(self.x_train, self.y_train, 
                            epochs=self.epochs, 
                            batch_size=self.batch_size,
                            validation_data=(self.x_valid, self.y_valid))
                            # callbacks=[early_stop, checkpoint])


        pred = model.predict(self.test_feature)
        loss = model.evaluate(self.test_feature,self.test_label)
        test_loss = loss
        wandb.log({"test_loss": test_loss})


        print(loss)

        plt.figure(figsize=(12, 9))
        plt.plot(self.test_label, label='actual')
        plt.plot(pred, label='prediction')
        plt.legend()
        plt.savefig(f'GRU_simple_{self.epochs}_{self.node}_{self.window_size}.png')
        # plt.show()
        model.save(f'GRU_simple_{self.epochs}_{self.node}_{self.window_size}.h5')

        factor.append("GRU_simple")
        val_loss_list.append(history.history['val_loss'][self.epochs-1])
        test_loss_list.append(loss)






        

            
class My_GRU_deep:
        # 생성자
    def __init__(self,nodes,feature,scaler_method,epochs, batch_size,window_size):
        self.nodes = nodes
        self.feature = feature
        self.scaler_method = scaler_method
        self.epochs = epochs
        self.batch_size = batch_size
        self.window_size = window_size

        df_price = pd.read_csv('./거시경제지표_kospi200_스케일링.csv', encoding='cp949')
        scaler = scaler_method
        scale_cols = df_price.columns

        df_scaled = scaler.fit_transform(df_price[scale_cols])

        df_scaled = pd.DataFrame(df_scaled)
        df_scaled.columns = scale_cols

        TEST_SIZE = 400

        train = df_scaled[:-TEST_SIZE]
        test = df_scaled[-TEST_SIZE:]

        feature_cols = self.feature
        label_cols = ['Close']

        self.train_feature = train[feature_cols]
        self.train_label = train[label_cols]

        self.test_feature = test[feature_cols]
        self.test_label = test[label_cols]

        self.train_feature, self.train_label = self.make_dataset(self.train_feature, self.train_label, self.window_size)
        self.test_feature, self.test_label = self.make_dataset(self.test_feature, self.test_label, self.window_size)
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.train_feature, self.train_label, test_size=0.2)



    def make_dataset(self,data, label, window_size):
        feature_list = []
        label_list = []
        for i in range(len(data) - window_size):
            feature_list.append(np.array(data.iloc[i:i+window_size]))
            label_list.append(np.array(label.iloc[i+window_size]))

        # print(feature_list)
        # print(label_list)
        return np.array(feature_list), np.array(label_list)
            
    def train(self):
        regression_GRU = Sequential()

        regression_GRU.add(GRU(units=self.nodes,activation="relu", return_sequences=True, input_shape = (self.train_feature.shape[1], self.train_feature.shape[2])))
        
        regression_GRU.add(GRU(units=self.nodes, activation="relu", return_sequences=True))

        regression_GRU.add(GRU(units=self.nodes, activation="relu", return_sequences=True))

        regression_GRU.add(GRU(units=self.nodes, activation="relu"))

        regression_GRU.add(Dense(units = 1))

        regression_GRU.compile(optimizer='adam', loss='mean_squared_error')
        GRU_history = regression_GRU.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, shuffle=False,validation_data=(self.x_valid, self.y_valid))

        pred = regression_GRU.predict(self.test_feature)
        loss = regression_GRU.evaluate(self.test_feature,self.test_label)
        test_loss = loss
        wandb.log({"test_loss": test_loss})

        print(loss)

        plt.figure(figsize=(12, 9))
        plt.plot(self.test_label, label='actual')
        plt.plot(pred, label='prediction')
        plt.legend()
        plt.savefig(f'GRU_deep_{self.epochs}_{self.nodes}_{self.window_size}.png')
        # plt.show()

        regression_GRU.save(f'GRU_deep_{self.epochs}_{self.nodes}_{self.window_size}.h5')

        factor.append("GRU_deep")
        val_loss_list.append(GRU_history.history['val_loss'][self.epochs-1])
        test_loss_list.append(loss)


class BackTesting:
    def __init__(self,model_name,model,money,feature,window_size,scaler_method,TEST_SIZE):
        self.money=money
        self.model = model
        self.model_name = model_name
        self.feature = feature
        self.window_size = window_size
        self.TEST_SIZE = TEST_SIZE
        
        
        self.my_profit = 0
        self.stock_num = 0
        self.left = 0
        self.position = "NONE"
        self.buy_price = 0

        column_name = ['name', 'code', 'Date','time', 'Open', 'High', 'Low', 'Close', 'Volume'] # 데이터의 열 지정

        self.inverse_data = pd.read_csv('./KODEX 인버스.csv', encoding='cp949')
        data = self.inverse_data.values.tolist()
        self.inverse_data = pd.DataFrame(data, columns=column_name) # 데이터에 열 지정

        self.leverage_data = pd.read_csv('./KODEX 레버리지.csv', encoding='cp949')
        data = self.leverage_data.values.tolist()
        self.leverage_data = pd.DataFrame(data, columns=column_name) # 데이터에 열 지정

        self.test_data = pd.read_csv('./거시경제지표_kospi200_스케일링_Date_inverse2_lever.csv', encoding='cp949')

        df_price = pd.read_csv('./거시경제지표_kospi200_스케일링_Date_inverse2_lever.csv', encoding='cp949')
        self.scaler = scaler_method
        self.df_price_drop = df_price.drop(['Date'], axis='columns')
        self.scale_cols = self.df_price_drop.columns

        self.df_scaled = self.scaler.fit_transform(self.df_price_drop[self.scale_cols])

        self.df_scaled = pd.DataFrame(self.df_scaled)
        self.df_scaled.columns = self.scale_cols
    
        train = self.df_scaled[:-self.TEST_SIZE]
        test = self.df_scaled[-self.TEST_SIZE:]


        # self.inverse_data = self.inverse_data[:420]
        # self.leverage_data = self.leverage_data[:420]

        # self.inverse_data = self.inverse_data[::-1]
        # self.leverage_data = self.leverage_data[::-1]

        # self.inverse_data = self.inverse_data.reset_index(drop=True)
        # self.leverage_data = self.leverage_data.reset_index(drop=True)


        

        feature_cols = self.feature
        label_cols = ['Close']

        self.test_feature = test[feature_cols]
        self.test_label = test[label_cols]
        self.test_date = df_price['Date'][-self.TEST_SIZE:]
        self.l_close = df_price['l_Close'][-self.TEST_SIZE:]
        self.i2_close = df_price['i2_Close'][-self.TEST_SIZE:]

        self.test_feature, self.test_label, self.test_date, self.l_close, self.i2_close = self.make_dataset(self.test_feature, self.test_label, self.test_date,self.window_size, self.l_close, self.i2_close)

   



    def make_dataset(self,data, label, date, window_size, l_close, i2_close):
        feature_list = []
        label_list = []
        date_list = []
        l_close_list = []
        i2_close_list = []
        for i in range(len(data) - window_size):
            feature_list.append(np.array(data.iloc[i:i+window_size]))
            label_list.append(np.array(label.iloc[i+window_size]))
            date_list.append(np.array(date.iloc[i+window_size]))
            l_close_list.append(np.array(l_close.iloc[i+window_size]))
            i2_close_list.append(np.array(i2_close.iloc[i+window_size]))



        # print(feature_list)
        # print(label_list)
        return np.array(feature_list), np.array(label_list), np.array(date_list), np.array(l_close_list), np.array(i2_close_list)


    def test(self):

        origin = pd.read_csv('./거시경제지표_kospi200_스케일X.csv', encoding='cp949')

        model = load_model(self.model)
        print(model.summary())
      
        pred = model.predict(self.test_feature)
        
        # pred_real = self.scaler.inverse_transform(self.df_scaled)

        # print(self.scale_cols)
        # print(pred_real[0])


        plt.figure(figsize=(12, 9))
        plt.plot(self.test_date,self.test_label, label='actual')
        plt.plot(pred, label='prediction')
        plt.legend()
        

        print(self.test_date)
        print(self.l_close)
        print(self.i2_close)
        for i in range(1,len(pred)):
            profit = (pred[i]/pred[i-1])
            # 비교를 함에 있어서 실제 주가로 해야할지 아니면 예상주가로 해야할지 불분명함
            # profit = (pred[i]/self.test_label[i-1])


            # print(self.test_date[i] , self.test_data['Date'][i])
      
            # print("before day : ",self.test_date[i-1],pred[i-1],"actually :",self.test_label[i-1])
            # print("next day : ",self.test_date[i],pred[i],"actually :",self.test_label[i])
            print(profit)
            if(profit > 1 and (self.position =="SHORT" or self.position == "NONE")):
                if(self.position == "NONE"):
                    plt.scatter(self.test_date[i-1], self.test_label[i-1], color="r", marker='^')
                    print("-"*10 + "LONG" + "-"*10)
                    print("tommorw : ",profit, "%")
                    print("before day : ",self.test_date[i-1],pred[i-1])
                    print("next day : ",self.test_date[i],pred[i])

                    print("Date : ",self.test_date[i-1])
                    print("Price : ",self.l_close[i-1])
                    
                    self.stock_num = self.money//(self.l_close[i-1]*1.00015)
                    self.buy_price = self.l_close[i-1]*1.00015
                    self.left = self.money - self.stock_num * (self.l_close[i-1]*1.00015)
                    self.position = "LONG"

                    print("Num : ",self.stock_num)
                    print("Left : ",self.left)
                    print("-"*14)
                elif(self.position == "SHORT"):
                    self.my_profit = self.i2_close[i-1] / self.buy_price
                    if(profit > 1.005):
                        print("-"*10 + "SWITCH to LONG" + "-"*10)
                        print("Date :",self.test_date[i-1])
                        print("num :",self.stock_num)
                        print("sell-price :",self.i2_close[i-1])
                        print("buy-price :",self.buy_price)

                        sell_price = self.stock_num * self.i2_close[i-1]
                        self.money = sell_price * 0.9935 + self.left

                        print("MONEY : ",self.money)

                        print("tommorw : ",profit, "%")
                        print("before day : ",self.test_date[i-1],pred[i-1])
                        print("next day : ",self.test_date[i],pred[i])

                        print("Date : ",self.test_date[i-1])
                        print("Price : ",self.l_close[i-1])
                        
                        self.stock_num = self.money//(self.l_close[i-1]*1.00015)
                        self.buy_price = self.l_close[i-1]*1.00015
                        self.left = self.money - self.stock_num * (self.l_close[i-1]*1.00015)
                        self.position = "LONG"

                        print("Num : ",self.stock_num)
                        print("Left : ",self.left)
                        plt.scatter(self.test_date[i-1], self.test_label[i-1], color="r", marker='^')

                        print("-"*14)
                    elif(self.my_profit < 0.95):
                        print("-"*10 + "LOSS-CUT" + "-"*10)
                        print("Date :",self.test_date[i-1])
                        print("num :",self.stock_num)
                        print("sell-price :",self.i2_close[i-1])
                        print("buy-price :",self.buy_price)

                        sell_price = self.stock_num * self.i2_close[i-1]
                        self.money = sell_price * 0.9935 + self.left

                        print("MONEY : ",self.money)

                        self.position = "NONE"
                        plt.scatter(self.test_date[i-1], self.test_label[i-1], color="b", marker='*')

                    
            elif(profit < 1 and (self.position =="LONG" or self.position == "NONE")):
                if(self.position == "NONE"):
                    plt.scatter(self.test_date[i-1], self.test_label[i-1], color="b", marker='v')
                    print("-"*10 + "SHORT" + "-"*10)
                    print(profit, "%")
                    print("before day : ",self.test_date[i-1],pred[i-1])
                    print("next day : ",self.test_date[i],pred[i])

                    print("Date : ",self.test_date[i-1])
                    print("Price : ",self.i2_close[i-1])
                    
                    self.stock_num = self.money//(self.i2_close[i-1]*1.00015)
                    self.buy_price = self.i2_close[i-1]*1.00015
                    self.left = self.money - self.stock_num * (self.i2_close[i-1]*1.00015)
                    self.position = "SHORT"

                    print("Num : ",self.stock_num)
                    print("Left : ",self.left)
                    print("-"*14)

                elif(self.position == "LONG"):
                    self.my_profit = self.l_close[i-1] / self.buy_price
                    if(profit < 0.995):
                        print("-"*10 + "SWITCH to SHORT" + "-"*10)
                        print("Date :",self.test_date[i-1])
                        print("num :",self.stock_num)
                        print("sell-price :",self.l_close[i-1])
                        print("buy-price :",self.buy_price)

                        sell_price = self.stock_num * self.l_close[i-1]
                        self.money = sell_price * 0.9935 + self.left

                        print("MONEY : ",self.money)

                        print("tommorw : ",profit, "%")
                        print("before day : ",self.test_date[i-1],pred[i-1])
                        print("next day : ",self.test_date[i],pred[i])

                        print("Date : ",self.test_date[i-1])
                        print("Price : ",self.i2_close[i-1])
                        
                        self.stock_num = self.money//(self.i2_close[i-1]*1.00015)
                        self.buy_price = self.i2_close[i-1]*1.00015
                        self.left = self.money - self.stock_num * (self.i2_close[i-1]*1.00015)
                        self.position = "SHORT"

                        print("Num : ",self.stock_num)
                        print("Left : ",self.left)
                        plt.scatter(self.test_date[i-1], self.test_label[i-1], color="b", marker='v')

                        print("-"*14)
                    elif(self.my_profit < 0.95):
                        print("-"*10 + "LOSS-CUT" + "-"*10)
                        print("Date :",self.test_date[i-1])
                        print("num :",self.stock_num)
                        print("sell-price :",self.l_close[i-1])
                        print("buy-price :",self.buy_price)

                        sell_price = self.stock_num * self.l_close[i-1]
                        self.money = sell_price * 0.9935 + self.left

                        print("MONEY : ",self.money)

                        self.position = "NONE"
                        plt.scatter(self.test_date[i-1], self.test_label[i-1], color="b", marker='*')


           

        plt.savefig(f'{self.model_name}_{self.window_size}_{round(self.money)}_{self.TEST_SIZE}.png')
        # plt.show()


class all_train:
    def __init__(nodes,feature_list , scaler_method ,epochs , batch_size , window_size, name):
        self.nodes = nodes
        self.feature_list = feature_list
        self.scaler_method = scaler_method
        self.epochs = epochs
        self.batch_size = batch_size
        self.window_size = window_size
        self.name = name


    def train():
        deep_LSTM = My_LSTM_deep(self.feature_list , self.scaler_method ,self.epochs , self.batch_size , self.window_size)
        deep_LSTM.train()

        simple_LSTM = My_LSTM_simple(self.nodes,self.feature_list , self.scaler_method ,self.epochs , self.batch_size , self.window_size)
        simple_LSTM.train()

        deep_GRU = My_GRU_deep(self.feature_list , self.scaler_method ,self.epochs , self.batch_size , self.window_size)
        deep_GRU.train()

        simple_GRU = My_GRU_simple(self.nodes,self.feature_list , self.scaler_method ,self.epochs , self.batch_size , self.window_size)
        simple_GRU.train()

        result={"model": factor, "val_loss":val_loss_list, "test_loss": test_loss_list}
        df = pd.DataFrame(result)
        df.to_csv(f'{self.name}.txt', sep = '\t', index = False)

        factor.clear()
        val_loss_list.clear()
        test_loss_list.clear()

def train(config=None):
    with wandb.init(config=config):
        config=wandb.config
        nodes = config.nodes
        feature_list = ['Open','High','Low','Volume','Close']
        scaler_method = MaxAbsScaler()
        epochs = config.epochs
        batch_size = config.batch_size
        window_size = config.window_size

        # Back = BackTesting("GRU_simple",10000000,['Open','High','Low','Volume','Close'],window_size,scaler_method)
        # Back.test()

        # deep_LSTM = My_LSTM_deep(nodes,feature_list , scaler_method ,epochs , batch_size , window_size)
        # deep_LSTM.train()

        simple_LSTM = My_LSTM_simple(nodes,feature_list , scaler_method ,epochs , batch_size , window_size)
        simple_LSTM.train()

        # deep_GRU = My_GRU_deep(nodes,feature_list , scaler_method ,epochs , batch_size , window_size)
        # deep_GRU.train()

        # simple_GRU = My_GRU_simple(nodes,feature_list , scaler_method ,epochs , batch_size , window_size)
        # simple_GRU.train()

        result={"model": factor, "val_loss":val_loss_list, "test_loss": test_loss_list}
        df = pd.DataFrame(result)
        # df.to_csv('using_all_result+close.txt', sep = '\t', index = False)

        factor.clear()
        val_loss_list.clear()
        test_loss_list.clear()

        # feature_list = ['Open','High','Low','Volume','Close']
        # scaler_method = MaxAbsScaler()

        # deep_LSTM = My_LSTM_deep(feature_list , scaler_method ,epochs , batch_size , window_size)
        # deep_LSTM.train()

        # simple_LSTM = My_LSTM_simple(nodes,feature_list , scaler_method ,epochs , batch_size , window_size)
        # simple_LSTM.train()

        # simple_GRU = My_GRU_simple(nodes,feature_list , scaler_method ,epochs , batch_size , window_size)
        # simple_GRU.train()

        # deep_GRU = My_GRU_deep(feature_list , scaler_method ,epochs , batch_size , window_size)
        # deep_GRU.train()

        # result={"model": factor, "val_loss":val_loss_list, "test_loss": test_loss_list}
        # df = pd.DataFrame(result)
        # # df.to_csv('using_index_result+close.txt', sep = '\t', index = False)

        # factor.clear()
        # val_loss_list.clear()
        # test_loss_list.clear()

        # feature_list = ['Open','High','Low','Volume','Close','macd','macdsignal','macdhist','rsi','MA5','MA20','MA60','MA224','cci']
        # scaler_method = MaxAbsScaler()

        # deep_LSTM = My_LSTM_deep(feature_list , scaler_method ,epochs , batch_size , window_size)
        # deep_LSTM.train()

        # simple_LSTM = My_LSTM_simple(nodes,feature_list , scaler_method ,epochs , batch_size , window_size)
        # simple_LSTM.train()

        # deep_GRU = My_GRU_deep(feature_list , scaler_method ,epochs , batch_size , window_size)
        # deep_GRU.train()

        # simple_GRU = My_GRU_simple(nodes,feature_list , scaler_method ,epochs , batch_size , window_size)
        # simple_GRU.train()

        # result={"model": factor, "val_loss":val_loss_list, "test_loss": test_loss_list}
        # df = pd.DataFrame(result)
        # # df.to_csv('using_index+stockinfo_result+close.txt', sep = '\t', index = False)

        # factor.clear()
        # val_loss_list.clear()
        # test_loss_list.clear()

        # feature_list = ['Crude Oil','10-Yr Bond','S&P 500','Dow Jones','Nasdaq','Nikkei 225','USD/KRW','EUR/KRW','JPY/KRW','Open','High','Low','Volume','Close']
        # scaler_method = MaxAbsScaler()

        # deep_LSTM = My_LSTM_deep(feature_list , scaler_method ,epochs , batch_size , window_size)
        # deep_LSTM.train()

        # simple_LSTM = My_LSTM_simple(nodes,feature_list , scaler_method ,epochs , batch_size , window_size)
        # simple_LSTM.train()

        # deep_GRU = My_GRU_deep(feature_list , scaler_method ,epochs , batch_size , window_size)
        # deep_GRU.train()

        # simple_GRU = My_GRU_simple(nodes,feature_list , scaler_method ,epochs , batch_size , window_size)
        # simple_GRU.train()

        # result={"model": factor, "val_loss":val_loss_list, "test_loss": test_loss_list}
        # df = pd.DataFrame(result)
        # # df.to_csv('using_index+biginfo_result+close.txt', sep = '\t', index = False)








if __name__ == "__main__":

    index = ['Open','High','Low','Volume','Close']
    all_feature = ['Open','High','Low','Volume','Close','Crude Oil','10-Yr Bond','S&P 500','Dow Jones','Nasdaq','Nikkei 225','USD/KRW','EUR/KRW','JPY/KRW','macd','macdsignal','macdhist','rsi','MA5','MA20','MA60','MA224','cci']
    index_big = ['Open','High','Low','Volume','Close','Crude Oil','10-Yr Bond','S&P 500','Dow Jones','Nasdaq','Nikkei 225','USD/KRW','EUR/KRW','JPY/KRW']
    index_small = ['Open','High','Low','Volume','Close','macd','macdsignal','macdhist','rsi','MA5','MA20','MA60','MA224','cci']
    feature_list = index
    scaler_method = MaxAbsScaler()
    window_size = [11,13,31,17,20]
    TEST_SIZE = 150
    model_name = 'only_index_gru_simple'
    model_path =    ['/home/user/GoogleDrive/index_predict/best5_only_index_gru_simple/GRU_best5/GRU_simple_90_122_35.h5',
    '/home/user/GoogleDrive/index_predict/best5_only_index_gru_simple/GRU_best5/GRU_simple_118_128_5.h5',
    '/home/user/GoogleDrive/index_predict/best5_only_index_gru_simple/GRU_best5/GRU_simple_181_34_18.h5',
    '/home/user/GoogleDrive/index_predict/best5_only_index_gru_simple/GRU_best5/GRU_simple_196_128_6.h5',
    '/home/user/GoogleDrive/index_predict/best5_only_index_gru_simple/GRU_best5/GRU_simple_197_90_28.h5']

    for i in range(len(model_path)):
        Back = BackTesting(model_name,model_path[i],10000000,feature_list,window_size[i],scaler_method,TEST_SIZE)
        Back.test()




'''
['/home/user/GoogleDrive/index_predict/best5_all_gru_simple/GRU_simple_63_100_19.h5',
'/home/user/GoogleDrive/index_predict/best5_all_gru_simple/GRU_simple_95_120_9.h5',
'/home/user/GoogleDrive/index_predict/best5_all_gru_simple/GRU_simple_129_41_39.h5,
'/home/user/GoogleDrive/index_predict/best5_all_gru_simple/GRU_simple_168_85_36.h5',
'/home/user/GoogleDrive/index_predict/best5_all_gru_simple/GRU_simple_170_110_49.h5']

['/home/user/GoogleDrive/index_predict/best5_all_gru_deep/GRU_deep_137_120_11.h5',
'/home/user/GoogleDrive/index_predict/best5_all_gru_deep/GRU_deep_143_124_13.h5',
'/home/user/GoogleDrive/index_predict/best5_all_gru_deep/GRU_deep_190_40_31.h5',
'/home/user/GoogleDrive/index_predict/best5_all_gru_deep/GRU_deep_199_104_17.h5',
'/home/user/GoogleDrive/index_predict/best5_all_gru_deep/GRU_deep_200_109_20.h5']

['/home/user/GoogleDrive/index_predict/best5_all_lstm_deep/LSTM_deep_151_59_13.h5',
    '/home/user/GoogleDrive/index_predict/best5_all_lstm_deep/LSTM_deep_162_128_5.h5',
    '/home/user/GoogleDrive/index_predict/best5_all_lstm_deep/LSTM_deep_168_99_8.h5',
    '/home/user/GoogleDrive/index_predict/best5_all_lstm_deep/LSTM_deep_177_127_44.h5',
    '/home/user/GoogleDrive/index_predict/best5_all_lstm_deep/LSTM_deep_185_74_42.h5']


    ['/home/user/GoogleDrive/index_predict/best5_only_index_gru_simple/GRU_best5/GRU_simple_90_122_35.h5',
    '/home/user/GoogleDrive/index_predict/best5_only_index_gru_simple/GRU_best5/GRU_simple_118_128_5.h5',
    '/home/user/GoogleDrive/index_predict/best5_only_index_gru_simple/GRU_best5/GRU_simple_181_34_18.h5',
    '/home/user/GoogleDrive/index_predict/best5_only_index_gru_simple/GRU_best5/GRU_simple_196_128_6.h5',
    '/home/user/GoogleDrive/index_predict/best5_only_index_gru_simple/GRU_best5/GRU_simple_197_90_28.h5']
'''
 
