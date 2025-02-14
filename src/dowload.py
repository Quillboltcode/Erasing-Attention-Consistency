import subprocess
import os
import shutil

def install_package(package):
    subprocess.check_call(["pip", "install", package])


def download_file(file_id, output_name):
    install_package("gdown")
    import gdown

    gdown.download(id=file_id, output=output_name, quiet=False)


def unzip_file(zip_file):
    import zipfile

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall()
    os.remove(zip_file)


def remove_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)

download_file("1yQRdhSnlocOsZA4uT_8VO0-ZeLXF4gKd","resnet50_ft_weight.pkl")
download_file("1EEx7qVCums-TM5fiblepgY70MDqIxbVz","resnet18_msceleb.pth")