import requests
def get_thumbnail_data(ss_id):
    url = f"https://apollo2.humanbrain.in/GW/getBrainThumbNailDetails/IIT/V1/SS-{ss_id}:-1:-1"

    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

#Example usage
session_id = "186"
biosample = "222"
thumbnails_data = get_thumbnail_data(100)
print(thumbnails_data[0].keys())


