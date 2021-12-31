import argparse
import falcon
from hparams import hparams, hparams_debug_string
import os
import numpy as np
from synthesizer import Synthesizer
from util import audio
import io
from flask import Flask, render_template, request, make_response


#
# html_body = '''<html><title>Demo</title>
# <style>
# body {padding: 16px; font-family: sans-serif; font-size: 14px; color: #444}
# input {font-size: 14px; padding: 8px 12px; outline: none; border: 1px solid #ddd}
# input:focus {box-shadow: 0 1px 2px rgba(0,0,0,.15)}
# p {padding: 12px}
# button {background: #28d; padding: 9px 14px; margin-left: 8px; border: none; outline: none;
#         color: #fff; font-size: 14px; border-radius: 4px; cursor: pointer;}
# button:hover {box-shadow: 0 1px 2px rgba(0,0,0,.15); opacity: 0.9;}
# button:active {background: #29f;}
# button[disabled] {opacity: 0.4; cursor: default}
# </style>
# <body>
# <form>
#   <input id="text" type="text" size="80"   placeholder="Enter your Text here">
#   <button id="button" name="synthesize">Speak</button>
# </form>
# <p id="message"></p>
# <audio id="audio" controls autoplay hidden></audio>
# <script>
# function q(selector) {return document.querySelector(selector)}
# q('#text').focus()
# q('#button').addEventListener('click', function(e) {
#   text = q('#text').value.trim()
#   if (text) {
#     q('#message').textContent = 'Synthesizing...'
#     q('#button').disabled = true
#     q('#audio').hidden = false
#     synthesize(text)
#   }
#   e.preventDefault()
#   return false
# })
# function synthesize(text) {
#   fetch('/synthesize?text=' + encodeURIComponent(text), {cache: 'no-cache'})
#     .then(function(res) {
#       if (!res.ok) throw Error(res.statusText)
#       return res.blob()
#     }).then(function(blob) {
#       q('#message').textContent = ''
#       q('#button').disabled = false
#       q('#audio').src = URL.createObjectURL(blob)
#       q('#audio').hidden = false
#     }).catch(function(err) {
#       q('#message').textContent = 'Error: ' + err.message
#       q('#button').disabled = false
#     })
# }
# </script></body></html>
# '''

# class UIResource:
#   def on_get(self, req, res):
#     res.content_type = 'text/html'
#     res.body = html_body
#
#
# class SynthesisResource:
#   def on_get(self, req, res):
#     if not req.params.get('text'):
#       raise Flask.HTTPBadRequest()
#     res.data = synthesizer.synthesize(req.params.get('text'))
#     res.content_type = 'audio/wav'


app = Flask(__name__, template_folder='template')
synthesizer = Synthesizer()




@app.route('/synthesize', methods=['GET'])
def synth():
  # response = make_response((synthesizer.synthesize(request.args.get("text"))).getvalue())
  splits = str(request.args.get("text")).replace(',', '.').split('. ', 10)
  print(splits)


  # result1=[]
  result=io.BytesIO()
  wave_f = []
  res1 = synthesizer.synthesize1(splits[0])
  for i in splits[1:]:
    # res = io.BytesIO()
    # response = make_response((synthesizer.synthesize(i).getvalue()))
    res = synthesizer.synthesize1(i)
    res1 = np.append(res1, res)  # res,res1 both are numpy array objects
    # print(res.shape)
    # wave_f.append(res)
    # result.write(res)
       # response = make_response(synthesizer.synthesize(request.args.get("text")))
  #    # l.append(make_response((synthesizer.synthesize(i)).getvalue())
    print(i)
    # final=io.BytesIO()
    # res.headers['Content-Type'] = 'audio/wav'
  audio.save_wav(res1, result)  # here res1 is numpy obj result is bytes.io obj

  result_final = make_response((result.getvalue()))   # result_final is response obj
  # result.append(res)
  # print(result)
  # response.headers['Content-Type'] = 'audio/wav'
  # response.headers['Content-Disposition'] = 'attachment; filename=sound.wav'
  # return response
  result_final.headers['Content-Type'] = 'audio/wav'
  return result_final


@app.route('/')
def UIRe():
  # return  UIRes
  return render_template('jn.html')


if __name__ == '__main__':

  from wsgiref import simple_server
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Full path to model checkpoint')
  parser.add_argument('--port', type=int, default=9000)
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')

  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  # os.environ["CUDA_VISIBLE_DEVICES"]=""     # loading model in cpu
  hparams.parse(args.hparams)
  print(hparams_debug_string())

  synthesizer.load(args.checkpoint)
  print('Serving on port %d' % args.port)
  # simple_server.make_server('0.0.0.0', args.port, api).serve_forever()
  app.run(host='0.0.0.0')
else:
  synthesizer.load(os.environ['CHECKPOINT'])
