import requests

url = "http://127.0.0.1:5000/predict"

audio_path = "../dataset/Audio_Speech_Actors_01-24/Actor_01/03-01-03-01-01-01-01.wav"

with open(audio_path, "rb") as audio:
    files = {
        "audio": audio
    }

    response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response:")
print(response.json())