from flask import Flask , render_template , url_for , request , redirect
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

app = Flask(__name__)

#ML function
def diabetes(a,b,c,d,e,f):

    #dataset
    df = pd.read_csv('diabetes.csv')
    df.drop(['Pregnancies','SkinThickness'],axis=1,inplace=True)
    X = df.drop(['Outcome'],axis=1)
    y = df['Outcome']

    #standardization
    scalar = StandardScaler()
    scalar.fit(X.values)
    Std_X = scalar.transform(X.values)
    X = Std_X

    #model
    classifier = svm.SVC(kernel='linear')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    classifier.fit(X_train,y_train)
    predictions = classifier.predict(X_test)
    accuracy = round(((accuracy_score(predictions,y_test)) * 100),2)

    #single_person
    # person_x = (148,72,0,33.6,0.627,50)
    person_x = (a,b,c,d,e,f)
    np_arr_x = np.array(person_x)
    np_arr_reshaped = np_arr_x.reshape(1,-1)
    std_data_x = scalar.transform(np_arr_reshaped)
    prediction_x = classifier.predict(std_data_x)
    output = prediction_x[0]
    return output



@app.route('/', methods=['POST', 'GET'])
def index():
    output = None
    if request.method == 'POST':
        a = request.form['a']
        b = request.form['b']
        c = request.form['c']
        d = request.form['d']
        e = request.form['e']
        f = request.form['f']

        if not a or not b or not c or not d or not e or not f:
            error_message = "Please fill in all fields"
            return render_template('index.html', error_message=error_message)

        output = diabetes(a,b,c,d,e,f)
        if output == 1 :
            output = 'You have Diabetes'
        else :
            output = "You doesn't have diabetes"
           
    
    return render_template('index.html',output=output)


if __name__ == "__main__":
    app.run(debug=True)