from django.urls import path
from . import views

app_name = 'stocks'

urlpatterns = [
    path('', views.home, name='home'),
    path('stock-prices/', views.stock_prices, name='stock_prices'),
    path('predictions/', views.predictions, name='predictions'),
    path('ajax/stock-data/', views.get_stock_data, name='get_stock_data'),
    path('ajax/predict/', views.predict_stock, name='predict_stock'),
]