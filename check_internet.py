

from urllib.request import urlopen

def internet_on():
    try:
        urlopen('http://google.com', timeout=7)
        print('true')
        return True
    except Exception as err: 
        return False

internet_on()