import certifi
print("Location is --->>",certifi.where())

import os
print(os.environ.get('REQUESTS_CA_BUNDLE'))