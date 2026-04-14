from django.shortcuts import render

# Create your views here.
import os
from django.shortcuts import render
from Employess.models import employeeRegistrationModel
from assets import *
from django.contrib import messages

def employeehome(request):
    return render(request, 'Employess/employeeHome.html', {})

# Create your views here.
def employeeRegister(request):
    if request.method=='POST':
        name=request.POST.get('name', '').strip()
        loginid=request.POST.get('loginid', '').strip()
        pswd=request.POST.get('pswd', '').strip()
        mobile=request.POST.get('mobile', '').strip()
        email=request.POST.get('email', '').strip()
        state=request.POST.get('state', '').strip()
        location=request.POST.get('location', '').strip()

        # Validation: Check for required fields
        if not all([name, loginid, pswd, mobile, email, state, location]):
            messages.error(request, 'All fields are required. Please fill in all the information.')
            return render(request, 'employeeRegistrations.html')

        # Validation: Check if loginid already exists
        if employeeRegistrationModel.objects.filter(loginid=loginid).exists():
            messages.error(request, f'Login ID "{loginid}" is already registered. Please choose a different one.')
            return render(request, 'employeeRegistrations.html')

        # Validation: Check if email already exists
        if employeeRegistrationModel.objects.filter(email=email).exists():
            messages.error(request, f'Email "{email}" is already registered. Please use a different email address.')
            return render(request, 'employeeRegistrations.html')

        # Validation: Check if mobile already exists
        if employeeRegistrationModel.objects.filter(mobile=mobile).exists():
            messages.error(request, f'Mobile number "{mobile}" is already registered. Please use a different number.')
            return render(request, 'employeeRegistrations.html')

        # If all validations pass, create and save the user
        try:
            ur = employeeRegistrationModel(
                name=name,
                loginid=loginid,
                password=pswd,
                email=email,
                state=state,
                location=location,
                mobile=mobile
            )
            ur.save()
            messages.success(request, 'You have been successfully registered! Please wait for admin approval.')
            return render(request, 'employeeRegistrations.html')
        except Exception as e:
            messages.error(request, f'An error occurred during registration: {str(e)}')
            return render(request, 'employeeRegistrations.html')
    else:
        return render(request, 'employeeRegistrations.html')

def employeeLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = employeeRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "Activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'Employess/employeeHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'employeeLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'employeeLogin.html', {})
        
import pandas as pd
from django.conf import settings 
#from .data_preprocessing import main, prediction_value # type: ignore
   
def dataset(request):
   
    path=r'media\insurance fraud claims.csv'
    df = pd.read_csv(path, nrows=100)
    df = df.to_html()
    return render(request, 'Employess/datasetreview.html', {'data': df})

def prediction(request):
    return render(request,'Employess/dataprediction.html')

from utility.data_procesing_improved import main as main_improved, prediction_value as predict_improved
from utility.data_procesing_max_accuracy import main as main_max, prediction_value as predict_max

def Classification_result(request):
    # Old improved XGBoost results
    acc_imp, prec_imp, rec_imp, bar_imp, conf_imp = main_improved()
    
    # New maximum accuracy results
    acc_max, prec_max, rec_max, bar_max, conf_max = main_max()
    
    return render(request, 'Employess/view_classification_result.html', context={
        'improved': {
            'accuracy': round(acc_imp * 100, 2),
            'precision': round(prec_imp * 100, 2),
            'recall': round(rec_imp * 100, 2),
            'barplot': bar_imp,
            'confusion': conf_imp,
        },
        'max': {
            'accuracy': round(acc_max * 100, 2),
            'precision': round(prec_max * 100, 2),
            'recall': round(rec_max * 100, 2),
            'barplot': bar_max,
            'confusion': conf_max,
        }
    })
    
def fruad_prediction(request):
    msg = ''
    if request.method == 'POST':
        policy_number=int(request.POST['policy_number'])
        collision_type = request.POST.get('collision_type')
        incident_severity = request.POST.get('incident_severity')
        authorities_contacted = request.POST.get('authorities_contacted')
        total_claim_amount = int(request.POST.get('total_claim_amount'))
        injury_claim =int( request.POST.get('injury_claim'))
        property_claim = int(request.POST.get('property_claim'))
        vehicle_claim = int(request.POST.get('vehicle_claim'))


        input_data = {
            'policy_number': int(request.POST['policy_number']),
            'age': int(request.POST['age']),
            'incident_type': request.POST.get('incident_type'),
            'collision_type': request.POST.get('collision_type'),
            'incident_severity': request.POST.get('incident_severity'),
            'authorities_contacted': request.POST.get('authorities_contacted'),
            'witnesses': int(request.POST.get('witnesses', 0)),
            'bodily_injuries': int(request.POST.get('bodily_injuries', 0)),
            'total_claim_amount': int(request.POST.get('total_claim_amount')),
            'injury_claim': int(request.POST.get('injury_claim')),
            'property_claim': int(request.POST.get('property_claim')),
            'vehicle_claim': int(request.POST.get('vehicle_claim')),
        }
    
        # Use the best model (max accuracy) for prediction
        result = predict_max(input_data)
        msg = 'fraud claim detected' if result == 'Y' else 'its a real claim'
    return render(request, 'Employess/dataprediction.html', {'result': msg})


