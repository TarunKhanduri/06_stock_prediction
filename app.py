from flask import Flask, render_template, request, url_for
import logic
import pandas as pd

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyse", methods=["GET", "POST"])
def analyse():

    if request.method == "POST":
        txt=request.form['text']
        text = 'dataset/'+txt+'.csv'
        df=pd.read_csv(text)
        df1,train_data,test_data,scaler=logic.preProcess(df)
        
        time_step = 100
        X_train, y_train = logic.create_dataset(train_data, time_step)
        X_test, y_test = logic.create_dataset(test_data, time_step)

        X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

        model=logic.stacked_lstm(X_train,y_train,X_test,y_test)

        train_predict,test_predict=logic.training(model,X_train,X_test,scaler)

        logic.plot(train_predict,test_predict,df1,scaler)

        logic.predict(model, test_data, df1, scaler)

    return render_template("predict.html",stock=txt)

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route("/back", methods=["GET", "POST"])
def back():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
