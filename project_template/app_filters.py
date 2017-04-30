from django import template

register = template.Library()

@register.filter(name="modulus")
def modulus(num,val):
    return num % val

