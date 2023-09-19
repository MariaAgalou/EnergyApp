from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Provider
import json
from .main import last_day_consumption, customers_statistics, customer_statistics, id_exists
from .forecasting import forecastingprocess


# Create your views here.


# Function for the login
def loginpage(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = authenticate(request, username=email, password=password)
        print(email, password, user)
        if user is not None:
            login(request, user)
            return redirect('overview')
        else:
            # Login failed!
            messages.error(request, 'Invalid username or password! Please try again!')
    
    return render(request, 'login.html')



# Function for the log out
def log_out(request):
    logout(request)

    template = loader.get_template('log_out.html')
    return HttpResponse(template.render())



# Function for the sign up
def signup(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        flname = request.POST.get('flname')
        username = request.POST.get('username')
        password = request.POST.get('password')
        phone = request.POST.get('phone')
        address = request.POST.get('address')
        date = request.POST.get('date')
        website = request.POST.get('website')


        try:

            user = Provider(email = email, flname = flname, username = username, password = password,
                     phone = phone, address = address, foundationdate = date, website = website)
            user.save()

        except Exception as e:
            print(e)

        
        if Provider.objects.filter(username = username).exists():
            # Username is not available
            return render(request, 'signup.html', {'username_taken': True})
        else:
            # Create user
            # my_user = User.objects.create_user(email, flname, username, password, phone, address, date, website)
            #my_user = User.objects.create_user(username, email, password)
            #my_user.save()

            # Create user
            #user = Provider(email = email, flname = flname, username = username, password = password,
             #        phone = phone, address = address, foundationdate = date, website = website)
            

            return redirect('overview')

    
    return render(request, 'signup.html')



# Function for the "Overview" option of the main menu
def overview(request):
    fig, lis = last_day_consumption()
    item1 = lis[0]
    item2 = round(lis[1], 3)
    item3 = round(lis[2], 3)
    item4 = lis[3]
    item5 = lis[4]
    item6 = lis[5]
    item7 = lis[6]
    item8 = lis[7]
    item9 = lis[8]


    context = {'graph': fig, 'item1' : item1, 'item2' : item2, 'item3' : item3, 'item4' : item4,
               'item5' : item5, 'item6' : item6, 'item7' : item7, 'item8' : item8, 'item9' : item9}

    return render(request, 'overview.html', context)



# Function for the "Records" option of the main menu
def records(request, steps=48, flag=False, day=True, week=False, month=False):
    
    # Call forecasting function
    rmse, ytest, fig = forecastingprocess(int(steps), flag, day, week, month)

    # Statistics
    low = ytest.min()
    high = round(ytest.max(), 4)
    total = round(sum(ytest), 3)
    estimated_cost = 1000
    actual_cost = 1120
    

    context = {'graph' : fig, 'rmse' : round(rmse, 3), 'estcost' : estimated_cost, 'actcost' : actual_cost, 
               'total' : total, 'low' : low, 'high' : high}

    return render(request, 'records.html', context)



# Function for the "Future Consumption" option of the main menu
def future_consumption(request, steps=48, flag=True, day=True, week=False, month=False):

    # Call forecasting function
    yhat, fig = forecastingprocess(int(steps), flag, day, week, month)

    # Statistics
    high = yhat.max()
    
    context = {'graph' : fig, 'high' : high}

    return render(request, 'future_consumption.html', context)


# Function for the "Customers" option of the main menu
def customers(request):

    arr = customers_statistics()

    # Display array
    json_records = arr.reset_index().to_json(orient ='records')
    data = []
    data = json.loads(json_records)
    context = {'d': data}
    
    return render(request, 'customers.html', context)



# Function about the statistics of one single customer
def customer(request, id):

    fig1, fig2, min, max, avg, lowtime, hightime = customer_statistics(id = id)

    context = {'graph1' : fig1, 'graph2' : fig2, 'min' : min, 'max' : max, 'avg' : avg, 
               'lowtime' : lowtime, 'hightime' : hightime}

    return render(request, 'customer.html', context)



# Function about the search of one specific customer
def search_customer(request):

    if request.method == 'POST':
        id = request.POST.get('id')

        # Check if ID is valid
        if id_exists(id):
            return redirect('customer', id = id)
        else:
            # Print error message if not
            messages.error(request, 'No customer found in the database with this ID! Try another ID!')

        
    return render(request, 'search_customer.html')



# Function for the "Profile" option of the avatar menu
def profile1(request):

    name = request.user.username
    
    try:
        #user_profile = Provider.objects.all().values()
        user_profile = Provider.objects.filter(username = name).values()
    except Provider.DoesNotExist:
        user_profile = None
        print("EXCEPTION")
    
    context = {'user_profile': user_profile}

    return render(request, 'profile1.html', context)



# Function for the modification of the profile 
def profile2(request):

    name = request.user.username
    
    try:
        user_profile = Provider.objects.filter(username = name).values()
        
    except Provider.DoesNotExist:
        user_profile = None
        print("EXCEPTION")

    if request.method == 'POST':
        email = request.POST.get('email')
        flname = request.POST.get('flname')
        username = request.POST.get('username')
        password = request.POST.get('password')
        phone = request.POST.get('phone')
        address = request.POST.get('address')
        date = request.POST.get('date')
        website = request.POST.get('website')


        # Updates - New data
        updates = {'email' : email, 'flname' : flname, 'username' : username, 'password' : password, 
                   'phone' : phone, 'address' : address, 'foundationdate' : date, 'website' : website}
        
        # Update data and Save
        user_profile.update(**updates)
        

        context= {'user_profile': user_profile}

        return render(request, 'profile1.html', context)
    

    
    context = {'user_profile': user_profile}
    
    return render(request, 'profile2.html', context)



# Function for the "Info" option of the avatar menu
def info(request):
    template = loader.get_template('info.html')
    return HttpResponse(template.render())



# Function for the "Help/Contact" option of the avatar menu
def help(request):
    template = loader.get_template('help.html')
    return HttpResponse(template.render())

