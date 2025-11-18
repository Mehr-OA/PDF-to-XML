'''
import os
from lxml import etree
import xml.etree.ElementTree as ET
from lxml.builder import E
import re
import requests
import os.path
from config_loader import CONFIG

def login_user():
    url = CONFIG.BASE_URL
    user = CONFIG.USER
    password = CONFIG.PASSWORD
    
    s = requests.Session()

    response = requests.get(url)
    cookie_value=''
    if response.status_code == 200:
        # Extract the CSRF token from cookies
        for cookie in response.cookies:
            print(cookie.value)
            cookie_value = cookie.value
    else:
        print("Failed to retrieve CSRF token")
        exit()

    # Set up headers for login with the CSRF token
    headers = {
        "X-XSRF-TOKEN": cookie.value
    }
    cookies = {
        "DSPACE-XSRF-COOKIE": cookie.value
    }

    # Data for the login request
    login_data = {
        "user": user,
        "password": password
    }
    
    login_url = CONFIG.LOGIN_ENDPOINT
    print(login_url)
    payload = {
        "user": user,
        "password": password
    }

    login_response = requests.post(login_url, data=payload, headers=headers, cookies=cookies)
    bearer_token=''
    if login_response.status_code == 200:
        print("Login successful!")
        bearer_token = login_response.headers.get('Authorization')
    else:
        print("Login failed")
        exit()

    # Check if the Bearer token was obtained
    if not bearer_token:
        print("Failed to retrieve Bearer token")
        exit()
    else:
        print(bearer_token)

    # Headers for the file upload, including the captured Bearer token
    upload_headers = {
        "X-XSRF-TOKEN": cookie_value,
        "Authorization": bearer_token
    }
'''

import requests
from io import BytesIO
from config_loader import CONFIG
from urllib.parse import urlparse

def get_authenticated_session():
    s = requests.Session()

    # --- Login (cookie + token) ---
    r = s.get(CONFIG.BASE_URL)
    cookie_value = ''
    for cookie in r.cookies:
        print('cookie', cookie.value)
        cookie_value = cookie.value
    
    cookies = {
        "DSPACE-XSRF-COOKIE": cookie_value
    }
    #xsrf = s.cookies.get("DSPACE-XSRF-COOKIE")
    #print(xsrf)
    payload = {"user": CONFIG.USER, "password": CONFIG.PASSWORD}
    headers = {"X-XSRF-TOKEN": cookie_value}
    r2 = s.post(CONFIG.LOGIN_ENDPOINT, data=payload, headers=headers)
    print(r2.status_code)
    r2.raise_for_status()
    
    token = r2.headers.get("Authorization")
    #upload_headers = {
        #"X-XSRF-TOKEN": cookie_value,
        #"Authorization": token
    #}
    
    parsed = urlparse(CONFIG.BASE_URL)
    domain = parsed.hostname            # "test.service.tib.eu"
    print(token)
    print(cookie_value)
    s.headers.update({"X-XSRF-TOKEN": cookie_value, "Authorization": token})
    s.cookies.set("DSPACE-XSRF-COOKIE", cookie_value, domain=domain, path="/renate/server")
    
    return s