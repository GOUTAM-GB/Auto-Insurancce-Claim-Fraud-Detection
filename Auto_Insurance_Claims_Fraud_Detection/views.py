from django.shortcuts import render


def index(request):
    return render(request,'index.html')
def employeeLogin(request):
    return render(request,'employeeLogin.html')
def employeeRegisterForm(request):
    return render(request,'employeeRegistrations.html')
def managementLogin(request):
    return render(request,'managementLogin.html')
