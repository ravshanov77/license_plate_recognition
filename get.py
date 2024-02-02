import requests

def downloadImg(link, fname):
    response = requests.get(link)

    if response.status_code == 200:
        with open(fname, 'wb') as file:
            file.write(response.content)
        print(f"Image downloaded successfully and saved as {fname}")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")

link = "https://www.dropbox.com/scl/fi/xncx2cuimd6jbliafhgux/photo_2023-12-14_15-09-00.jpg?rlkey=jypf9f17rt604fp0if5w9oki6&dl=0"
fname = "car.jpg"
downloadImg(link, fname)
