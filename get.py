import requests

def download_image_from_dropbox(dropbox_link, local_filename):
    # Make a GET request to the Dropbox link
    response = requests.get(dropbox_link)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Open a local file in binary mode and write the content of the response
        with open(local_filename, 'wb') as file:
            file.write(response.content)
        print(f"Image downloaded successfully and saved as {local_filename}")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")

# Example usage:
dropbox_link = "https://www.dropbox.com/scl/fi/xncx2cuimd6jbliafhgux/photo_2023-12-14_15-09-00.jpg?rlkey=jypf9f17rt604fp0if5w9oki6&dl=0"
local_filename = "car.jpg"
download_image_from_dropbox(dropbox_link, local_filename)
