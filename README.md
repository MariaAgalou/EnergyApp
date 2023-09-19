# EnergyApp
### **Real-time visualization and prediction system for electrical energy consumption data**

This project is part of my thesis for my bachelor degree on Informatics and Telematics.

Energy App is a web application that was developed in order to be a useful tool in the hands of an electricity provider. More specifically, by using this application an electricity provider is able to:

1) See some statistics about the electrical energy consumption of the last 24 hours
2) See some statistics and information through special diagrams about his customers
3) Predict the electrical energy consumption in the future using machine learning algorithms
4) See how successful and accurate the predictions that this application makes are

Framework Django as well as Javascript, HTML & CSS were used for the front-end and back-end part of the application, while MongoDB was used as a database. The machine learning algorithms that were tested were ARIMA, LSTM, GRU and Transformers. The whole project is written in Python.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**To run this app:**

- Be sure that you have Python, Django, MongoDB and all the necessary modules installed in your system.
- Download the project and save it in a directory in your filesystem.
- Make sure you have MongoDB server running.
- Connect to MongoDB.
- Open the project inside an IDE and run the command "py manage.py runserver".
- Open a web browser, go to http://127.0.0.1:8000/ and use the application.
