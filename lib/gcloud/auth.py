from pathlib import Path
from enum import Enum
from typing import NamedTuple, List

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

CREDS_DIR = Path('.creds/')
CREDS_DIR.mkdir(exist_ok=True)

gitignore = CREDS_DIR / '.gitignore'
if not gitignore.is_file():
    with open(gitignore, 'w+') as doc:
        doc.write('*\n')

class CredFiles(NamedTuple):
    creds: Path
    token: Path

def get_credentials(scopes: List[str]) -> Credentials:
    creds = CREDS_DIR / 'secret.json'
    token = CREDS_DIR / 'token.json'

    if not creds.is_file():
        raise Exception('missing credentials: ' + str(creds))

    creds_file, token_file = CredFiles(creds, token)

    creds = None

    # The file token_file stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if token_file.is_file():
        creds = Credentials.from_authorized_user_file(token_file, scopes)

    # If there are no (valid) credentials available, let the user log in.
    if creds is None or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                creds_file, scopes
            )
            creds = flow.run_local_server(port=0)

        with open(token_file, 'w', encoding='utf-8') as token:
            token.write(creds.to_json())

    return creds
