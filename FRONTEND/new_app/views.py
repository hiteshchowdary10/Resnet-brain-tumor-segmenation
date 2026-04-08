from django.shortcuts import render
from django.http import HttpResponse
from django.core.exceptions import ValidationError
from .models import predict_tumor
from PIL import Image


# -------------------------------
# Validate uploaded image
# -------------------------------
def validate_image(img):
    try:
        Image.open(img).verify()
    except Exception:
        raise ValidationError("Uploaded file is not a valid image.")


# -------------------------------
# Home Page
# -------------------------------
def home(request):
    return render(request, 'index.html')


# -------------------------------
# Login Page
# -------------------------------
def input(request):
    file_name = 'account.txt'
    name = request.POST.get('name')
    password = request.POST.get('password')

    with open(file_name, 'r') as file:
        account_list = [line.split() for line in file]

    for account in account_list:
        if account[0] == name and account[1] == password:
            return render(request, 'input.html')

    return HttpResponse('Wrong Password or Name', content_type='text/plain')


# -------------------------------
# Prediction Output
# -------------------------------
def output(request):
    if request.method != "POST":
        return HttpResponse("Invalid request method.", content_type="text/plain")

    if 'file' not in request.FILES:
        return HttpResponse("No file uploaded.", content_type="text/plain")

    img = request.FILES['file']

    # Validate uploaded image
    try:
        validate_image(img)
    except ValidationError as e:
        return HttpResponse(str(e), content_type="text/plain")

    # Perform prediction
    result = predict_tumor(img)

    print("Prediction Result:", result)

    return render(request, 'output.html', {
        'out': result["label"],
        'confidence': round(result["confidence"], 2),
        'class_index': result["class_index"]
    })