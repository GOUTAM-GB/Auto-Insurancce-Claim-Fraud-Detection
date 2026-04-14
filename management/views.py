# Create your views here.
from django.shortcuts import render
from django.contrib import messages
import pandas as pd

from Employess.models import employeeRegistrationModel

def managementHome(request):
    return render(request, 'Management/managementhome.html')

# Create your views here.
def managementLoginCheck(request):
    if request.method=="POST":
       loginid=request.POST['loginid']
       pswd=request.POST['pswd']
       if loginid=='admin' and pswd=='admin':
           
           return render(request,'Management/managementhome.html')
       else:
            messages.error(request, 'Please enter details carefully')

            return render(request,'ManagementLogin.html')

def employeeDetails(request):
    ud=employeeRegistrationModel.objects.all()
    print(ud)
    return render(request,'Management/employeeDetais.html',context={'ud':ud})

def datasetdetails(request):
    path=r'media\insurance fraud claims.csv'
    data=pd.read_csv(path, nrows=100).to_html()
    return render(request,'Management/datasetreview.html',context={'data':data})


def updateEmployeeStatus(request):
    loginid=request.GET.get('loginid')
    if loginid:
        try:
            usu=employeeRegistrationModel.objects.get(loginid=loginid)
            if usu.status == 'waiting':
                usu.status='Activated'
                usu.save()
        except employeeRegistrationModel.DoesNotExist:
            messages.error(request, 'Employee not found.')

    ud = employeeRegistrationModel.objects.all()
    return render(request,'Management/employeeDetais.html',context={'ud':ud})


def DeleteUsers(request):
    uid = request.GET.get('uid')
    if uid:
        try:
            user = employeeRegistrationModel.objects.get(id=uid)
            user.delete()
            messages.success(request, 'Employee deleted successfully.')
        except employeeRegistrationModel.DoesNotExist:
            messages.error(request, 'Employee not found.')
    else:
        messages.error(request, 'Missing user id for deletion.')

    ud = employeeRegistrationModel.objects.all()
    return render(request,'Management/employeeDetais.html', context={'ud':ud})

    


