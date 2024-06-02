import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import model


class RequestHandler(BaseHTTPRequestHandler):
    _loaded_model = None
    _vectorizer = None
    def __init__(self, request, client_address, server):
        if RequestHandler._loaded_model is None:
            RequestHandler._loaded_model = model.load_model()
        if RequestHandler._vectorizer is None:
            RequestHandler._vectorizer = model.load_model('vectorizer.pkl')
        super().__init__(request, client_address, server)
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, hx-current-url, hx-request, hx-target, hx-trigger')
        self.end_headers()
    def do_POST(self):
        if self.path == '/predict':
            # Get the email from the input box on the site
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            print("Post Data:")
            print(post_data)

            # htmx form
            from urllib.parse import parse_qs
            data = parse_qs(post_data)
            print(data)
            email_text = data[b'email'][0].decode('utf-8')

            # Vanilla JS
            # data = json.loads(post_data)
            # print(data)
            # email_text = data['email']

            # load model
            prediction = model.model_predict(self._loaded_model, email_text, self._vectorizer)
            response_data = {'result': prediction}
            response_json = json.dumps(response_data)
            print(f'Prediction: {prediction}')
            self.send_response(200, 'Successfully used predict path!')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(response_json.encode())
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port=5555):
    server_address = ("", port)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f'Starting server on port {port}...')
    httpd.serve_forever()


if __name__ == '__main__':
    run_server()
