import sys
import json
import requests

ROUTER_API_URL = "http://localhost:8080/process/"  # Port 8080 in docker-compose.yml


def start_tutor(question):
    try:
        response = requests.post(ROUTER_API_URL, json={"text": question})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Error contacting router_api: {e}"}


def main():
    response = {"message": "error"}
    try:
        if len(sys.argv) > 1:
            question = sys.argv[1]
        else:
            question = "Default Question"

        message_json = start_tutor(question)
        message = message_json.get("citation", "No message generated.")

        response.update({"message": message})
    except Exception as e:
        response.update({"message": json.dumps(str(e).replace('"', '\\"').replace('\n', '\\n'))})
    print(json.dumps(response))


if __name__ == "__main__":
    main()
