from django.urls import path
from . import views


app_name = 'learning_modules'

urlpatterns = [
    path('', views.index, name = 'index'),
    path('index/', views.index, name = 'index'),
    path('introduction/', views.introduction, name = 'introduction'),
    path('compare/', views.compare, name = 'compare'),
    path('compare-info/', views.compare_info_form, name='compare_info_form'),
    path('whatIsInvesting/', views.whatIsInvesting, name = 'whatIsInvesting'),
    path('riskAndDiversification/', views.riskAndDiversification, name = 'riskAndDiversification'),
    path('fixedIncome/', views.fixedIncome, name = 'fixedIncome'),
    path('stocks/', views.stocks, name = 'stocks'),
    path('mutualFunds/', views.mutualFunds, name = 'mutualFunds'),

]
