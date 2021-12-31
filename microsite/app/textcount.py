from flask import  Flask,render_template,request,make_response

app = Flask(__name__,template_folder='template')

html_body = '''<html><title>Demo</title>
<style>
body {padding: 16px; font-family: sans-serif; font-size: 14px; color: #444}
input {font-size: 14px; padding: 8px 12px; outline: none; border: 1px solid #ddd}
input:focus {box-shadow: 0 1px 2px rgba(0,0,0,.15)}
p {padding: 12px}
button {background: #28d; padding: 9px 14px; margin-left: 8px; border: none; outline: none;
        color: #fff; font-size: 14px; border-radius: 4px; cursor: pointer;}
button:hover {box-shadow: 0 1px 2px rgba(0,0,0,.15); opacity: 0.9;}
button:active {background: #29f;}
button[disabled] {opacity: 0.4; cursor: default}
</style>
<body>
<form>
  <input id="text" type="text" size="40" placeholder="Enter Text">
  <button id="button" name="synthesize">Speak</button>
</form>
<p id="message"></p>
<audio id="audio" controls autoplay hidden></audio>
<script>
function q(selector) {return document.querySelector(selector)}
q('#text').focus()
q('#button').addEventListener('click', function(e) {
  text = q('#text').value.trim()
  if (text) {
    q('#message').textContent = 'Synthesizing...'
    q('#button').disabled = true
    q('#audio').hidden = false
    synthesize(text)
  }
  e.preventDefault()
  return false
})
function synthesize(text) {
  fetch('/synthesize?text=' + encodeURIComponent(text), {cache: 'no-cache'})
    .then(function(res) {
      if (!res.ok) throw Error(res.statusText)
      return res.blob()
    }).then(function(blob) {
      q('#message').textContent = ''
      q('#button').disabled = false
      q('#audio').src = URL.createObjectURL(blob)
      q('#audio').hidden = false
    }).catch(function(err) {
      q('#message').textContent = 'Error: ' + err.message
      q('#button').disabled = false
    })
}
</script></body></html>
'''


@app.route('/synthesis',methods=['GET'])
def synth():
  response = make_response(synthesizer.synthesize(request.args.get("text")))
  # response1=request.args.get("text")
  # buf.close()
  splits = response1.split('. ', 4)
  for i in splits:
      print(i)
  # response.headers['Content-Disposition'] = 'attachment; filename=sound.wav'
  return response
  # Response.data=synthesizer.synthesize(request.args.get("text"))
  # Response.content_type= 'audio/wav'
  # return

@app.route('/')
def UIRe():
  # return  UIRes
  return render_template('jn.html')



if __name__ == '__main__':

    text="We will make sure India gets it due. Whatever we have been getting is back end money, We will speak to the ICC and take the matter forward"
    splits=text.replace(',','.').split('. ',4)
    for i in splits:
         print(i)

    app.run(debug=True)

