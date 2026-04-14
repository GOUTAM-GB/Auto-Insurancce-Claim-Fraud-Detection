"""
URL configuration for Auto_Insurance_Claims_Fraud_Detection project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from Employess import views as em
from management import views as mv

from Auto_Insurance_Claims_Fraud_Detection import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('employeeLogin', views.employeeLogin, name='employeeLogin'),
    path('employeeRegisterForm', views.employeeRegisterForm, name='employeeRegisterForm'),
    path('managementLogin', views.managementLogin, name='managementLogin'),

    # Employees URLs
    path('employeeRegister', em.employeeRegister, name='employeeRegister'),
    path('employeeLoginCheck', em.employeeLoginCheck, name='employeeLoginCheck'),
    path('employeeHome/', em.employeehome, name='employeehome'), # Keep this one
    path('dataset', em.dataset, name='dataset'),
    path('Classification_result', em.Classification_result, name='Classification_result'),
    path('prediction', em.prediction, name='prediction'),
    path('fruad_predict', em.fruad_prediction, name='fruad_prediction'),

    # Management URLs
    path('managementLoginCheck', mv.managementLoginCheck, name='managementLoginCheck'),
    path('employeeDetails', mv.employeeDetails, name='employeeDetails'),
    path('datasetdetails', mv.datasetdetails, name='datasetdetails'),
    path('updateEmployeeStatus', mv.updateEmployeeStatus, name='updateEmployeeStatus'),
    path('DeleteUsers/', mv.DeleteUsers, name='DeleteUsers'),
    path('managementHome', mv.managementHome, name='managementHome')
]
