
import RPi.GPIO as GPIO
from http.server import BaseHTTPRequestHandler, HTTPServer

GPIO.setmode(GPIO.BCM)
GPIO.setup(20, GPIO.OUT)
Request = None

class RequestHandler_httpd(BaseHTTPRequestHandler):
  def do_GET(self):
    global Request
    messagetosend = bytes('',"utf")
    self.send_response(200)
    self.send_header('Content-Type', 'text/plain')
    self.send_header('Content-Length', len(messagetosend))
    self.end_headers()
    self.wfile.write(messagetosend)
    Request = self.requestline
    Request = Request[5 : int(len(Request)-9)]
    print(Request)
    if Request == 'on':
      GPIO.output(20,True)
    if Request == 'off':
      GPIO.output(20,False)
    if Request == 'records':
      messagetosend = bytes((open('ML_Log.log','r')),"utf")
      self.send_response(200)
      self.send_header('Content-Type', 'text/plain')
      self.send_header('Content-Length', len(messagetosend))
      self.end_headers()
      self.wfile.write(messagetosend)
    return


server_address_httpd = ('10.8.86.5',8080)
httpd = HTTPServer(server_address_httpd, RequestHandler_httpd)
httpd.serve_forever()
GPIO.cleanup()