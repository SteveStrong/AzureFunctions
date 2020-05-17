import logging

import azure.functions as func
from NLPData import NLPData
from NLPEngine import NLPEngine


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    nlp = NLPEngine();
    nlp.load("version1")

    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        text = f"Hello {name} The Veteran did not have a psychiatric disorder in service that was unrelated to the use of drugs."
    
        result = nlp.predict(text, print=True)
    
        return func.HttpResponse(
            body=result,
            status_code=200
        )
    else:
        return func.HttpResponse(
             "Please pass a name param  on the query string or in the request body",
             status_code=400
        )
