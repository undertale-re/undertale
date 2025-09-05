import requests
import json

headers = {"Content-Type": "application/json"}
def post_http_request_outline(prompt, addr, schema, max_tokens=1000):
    data = {
        "prompt": prompt,
        "schema": schema,
        "max_tokens": max_tokens
       }

    # Request response
    response = requests.post(addr, headers=headers, json=data, stream=True)
    return response

def get_response_outline(response):
    if response.status_code != 400:
        try:
            data = json.loads(response.content)
        except:
            print(response.content)
            raise ValueError()
        if 'text' in data:
            output = data['text']
        else:
            print(data)
            raise ValueError()
    else:
        output = "400 response"
    return output

required = ["summary"]
schema = {
  "title": "source code summarization",
  "type": "object",
  "properties": {
    "summary": {"type": "string"}
  },
  "required": required
}

def validate_outline(output):
    return True, output
    # response = output.split('```')[2].strip()
    # try:
    #     row = json.loads(response)
    #     return True, row
    # except:
    #     try:
    #         return True, row
    #     except:
    #         return False, response