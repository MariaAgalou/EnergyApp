from django.urls import path
from . import views

# Definition of URLs
urlpatterns = [
    path('', views.loginpage, name='login'),
    path('signup', views.signup, name='signup'),
    path('overview', views.overview, name='overview'),
    path('records', views.records, name='records'),
    path('future_consumption', views.future_consumption, name='future_consumption'),
    path('customers', views.customers, name='customers'),
    path('customer/<id>', views.customer, name='customer'),
    path('search_customer', views.search_customer, name='search_customer'),
    path('profile1', views.profile1, name='profile1'),
    path('profile2', views.profile2, name='profile2'),
    path('info', views.info, name='info'),
    path('help', views.help, name='help'),
    path('log_out', views.log_out, name='log_out'),
    path('records/<steps>/<flag>/<day>/<week>/<month>/', views.records, name='records'),
    path('future_consumption/<steps>/<flag>/<day>/<week>/<month>/', views.future_consumption, name='future_consumption')
]
