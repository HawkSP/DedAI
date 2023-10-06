import os
from dotenv import load_dotenv

# Load variables from .env file if it exists
load_dotenv()

# Load sensitive information from environment variables
CLIENT_ID = os.getenv('CORTEX_CLIENT_ID')
CLIENT_SECRET = os.getenv('CORTEX_CLIENT_SECRET')
HEADSET_ID = os.getenv('CORTEX_HEADSET_ID')
CORTEX_TOKEN = os.getenv('CORTEX_CORTEX_TOKEN')

# Other configuration variables
# You can add other configuration variables here as your project grows
