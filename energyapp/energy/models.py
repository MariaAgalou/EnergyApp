from django.db import models

# Create your models here.
class Provider(models.Model):

    email = models.CharField(max_length=200, unique=True)
    flname = models.CharField(max_length=100)
    username = models.CharField(max_length=200, unique=True)
    password = models.CharField(max_length=50)
    phone = models.CharField(max_length=10)
    address = models.CharField(max_length=250)
    foundationdate = models.DateField()
    website = models.CharField(max_length=500)


    def __str__(self):
        return self

