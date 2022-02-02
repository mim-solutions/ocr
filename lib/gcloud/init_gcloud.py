from lib.gcloud.auth import get_credentials


if __name__ == '__main__':
    #scopes = ['https://www.googleapis.com/auth/drive.readonly']
    scopes = ['https://www.googleapis.com/auth/cloud-vision']
    get_credentials(scopes)
