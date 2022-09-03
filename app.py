# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('ET_model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index1.html')



@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':
        online_order=request.form['online_order']
        if (online_order=='Yes'):
            online_order=1
        else:
            online_order=0    
        book_table=request.form['book_table']
        if (book_table=='Yes'):
            book_table=1
        else:
            book_table=0
        votes  = request.form['votes'] 
        rest_type=request.form['rest_type']
        if (rest_type=="Casual Dining"):
            rest_type = 1
        else:
            rest_type=0
        dish_liked = request.form['dish_liked']
        if (dish_liked=="Masala Dosa"):
            dish_liked=1
        else:
            dish_liked=0
        cuisine = request.form['cuisines']
        if (cuisine=='North Indian'):
            cuisine = 1
        else:
            cuisine = 0
        cost =float(request.form['cost'])
        review_list = request.form['reviews_list']
        if (review_list=='Great Food'):
            review_list=1
        else:
            review_list=0
        Type =request.form['type']  
        if (Type=="Buffet"):
            Type=1
        else:
            Type=0
        
        x = [online_order,book_table,votes,rest_type,dish_liked,cuisine,cost,review_list,Type]
        
        prediction=model.predict([x])
        output=round(prediction[0],2)
        if output<0:
            return render_template('index1.html',prediction_texts="Sorry no rating available.")
        else:
            return render_template('index1.html',prediction_text="Restaurant Rating is: {}".format(output))
    else:
        return render_template('index1.html')

if __name__=="__main__":
    app.run(debug=True)
