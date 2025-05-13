import requests

# the api app (fastapi) is in src folder. so use the following to run server
# uvicorn app:app --app-dir src

# once running following will be the URL
URL = "http://127.0.0.1:8000"



apis = ["prediction",
        "model_score",
        "statistics",
        "diagnostics"]

responses = []
for api in apis:
    print("URL : ", URL + f"/{api}")
    responses.append(requests.get(URL + f"/{api}"))

#write the responses to your workspace


with open('apireturns.txt', "w") as f:
    f.write("API Responses\n")
    for idx, response in enumerate(responses):
        f.write("\n ------------------------------------- \n")
        f.write(f"result from {apis[idx]} : status : {response}\n")

        f.write(str(response.json()))
        f.write("\n")