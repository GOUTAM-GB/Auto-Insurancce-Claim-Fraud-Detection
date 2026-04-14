from django.db import models

# Create your models here.
from django.db import models

from django.core.validators import RegexValidator, EmailValidator, MinLengthValidator

class employeeRegistrationModel(models.Model):
    name = models.CharField(max_length=100)
    
    # Use RegexValidator for login ID
    loginid = models.CharField(
        unique=True,
        max_length=100,
        validators=[
            RegexValidator(
                regex=r'^[a-zA-Z0-9]*$',  # Allow only alphanumeric characters
             
            ),
        ]
    )
    
    # Password with a minimum length validator
    password = models.CharField(
        max_length=100,
        validators=[
            RegexValidator(
                regex=r'^(?=.*[A-Z])(?=.*\d)(?=.*[\W_])[A-Z\w\W_]{6,}$',
            ),
        ]
    )
    
    # Mobile number validator
    mobile = models.CharField(
        unique=True,
        max_length=15,  # Adjusted for typical mobile number lengths
        validators=[
            RegexValidator(
                regex=r'^\+?1?\d{9,15}$',  # Allows international formats, adjust as needed
            ),
        ]
    )

    # Email field with EmailValidator
    email = models.EmailField(
        unique=True,
        max_length=100,
        validators=[EmailValidator(message='Enter a valid email address.')]
    )
    
    location = models.CharField(max_length=100)
    state = models.CharField(max_length=100)

    # Status field - you can add choices if required
    status = models.CharField(
        max_length=100,
        default='waiting'  # Set default status to 'waiting'
    )


    def __str__(self):
        return self.name
