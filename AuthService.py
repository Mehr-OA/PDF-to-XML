import requests
from urllib.parse import urlparse
from config_loader import CONFIG


def get_authenticated_session():
    session = requests.Session()

    try:
        # Get XSRF cookie
        response = session.get(CONFIG.BASE_URL)
        response.raise_for_status()

        cookie_value = response.cookies.get("DSPACE-XSRF-COOKIE")

        payload = {
            "user": CONFIG.USER,
            "password": CONFIG.PASSWORD
        }

        headers = {"X-XSRF-TOKEN": cookie_value}

        login_response = session.post(CONFIG.LOGIN_ENDPOINT, data=payload, headers=headers)

        # Handle wrong credentials
        if login_response.status_code in [401, 403]:
            print("Login failed: wrong email or password")
            return None

        login_response.raise_for_status()

        token = login_response.headers.get("Authorization")

        if not token:
            print("Login failed: no authorization token received")
            return None

        # Setup session
        parsed = urlparse(CONFIG.BASE_URL)
        domain = parsed.hostname

        session.headers.update({
            "X-XSRF-TOKEN": cookie_value,
            "Authorization": token
        })

        session.cookies.set(
            "DSPACE-XSRF-COOKIE",
            cookie_value,
            domain=domain,
            path="/renate/server"
        )

        print("Successfully logged in")
        return session

    except requests.exceptions.ConnectionError:
        print("Cannot connect to server")

    except requests.exceptions.Timeout:
        print("Request timed out")

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")

    return None