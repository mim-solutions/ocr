from pathlib import Path
import io

from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import build

from lib.gcloud.auth import ServiceAccount, get_credentials
from lib.url import Url


class GoogleDriveUrl(Url):

    def __init__(self, urlstring: str) -> None:
        super().__init__()

        assert self.scheme == 'https'
        assert self.netloc == 'drive.google.com'

    def get_id(self):
        parts = self.path.split('/')
        # print(parts)
        assert parts[0] == ''
        assert parts[1] == 'file'
        assert parts[2] == 'd'
        assert parts[4] == 'view'
        assert self.query == 'usp=sharing'

        return parts[3]

class GoogleDocsUrl(Url):

    def __init__(self, urlstring: str) -> None:
        super().__init__()

        assert self.scheme == 'https'
        assert self.netloc == 'docs.google.com'
    def get_id(self):
        parts = self.path.split('/')
        # print(parts)
        assert parts[0] == ''
        assert parts[1] == 'spreadsheets'
        assert parts[2] == 'd'
        assert parts[4] == 'edit'

        return parts[3]




def get_gdrive_service():
    scopes = ['https://www.googleapis.com/auth/drive.readonly']
    creds = get_credentials(ServiceAccount.MEDIA_MONITORING, scopes)
    service = build('drive', 'v3', credentials=creds)
    return service


# file_id is the unique ID of the file in google drive
def save_file(file_id: str, dest_file: Path, overwrite: bool = False) -> None:
    if not overwrite and dest_file.is_file():
        raise Exception('dest file already exists')

    service = get_gdrive_service()

    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))

    with open(dest_file, 'wb+') as doc:
        doc.write(fh.getbuffer())
