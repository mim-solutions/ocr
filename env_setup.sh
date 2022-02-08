#!/bin/bash

set -e #exit immediately after one of the commands failed

pip install --upgrade pip

pip install google-auth-oauthlib \
            google-cloud-vision \
            ipykernel \
            notebook \
            opencv-python \
            pytesseract \
            scipy \
            