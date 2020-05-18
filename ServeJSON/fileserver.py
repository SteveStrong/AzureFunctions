import logging
import json
import sys

import azure.functions as func


class PayloadWrapper:
    def wrapList(self, payload, message=''):
        result = {
            'hasErrors': len(message) > 0,
            'message': message,
            'payloadCount': len(payload),
            'payload': payload,
        }
        return json.dumps(result, indent=4, default=str)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    try:

        # ...using the json file loader to translate the json data...
        file = open('ServeJSON/data/tasks.json')    
        data = json.load(file)

        pw = PayloadWrapper()
        return func.HttpResponse(
            body=pw.wrapList(data),
            status_code=200
        )
    except:
        pw = PayloadWrapper()
        message = sys.exc_info()[0]
        return func.HttpResponse(
            body= pw.wrapList([],'Please pass a name on the query string or in the request body'),
             status_code=400
        )
