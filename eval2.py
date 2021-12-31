import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer


sentences = [
  'Take a left turn after 100 m. and you will be on Mahatma Gandhi Marg',
  'go forward for 200 m. and take a left turn',
  'You should reach your destination by 2:35',
  'You are on the fastest route despite usual traffic',
  'In 1000 ft. use the right lane to take eastern peripheral expressway',
  'In 100 ft. merge on to yamuna expressway',
  'In 100 m. make a left turn on Guru Ravidas Marg',
  'Continue on yamuna expressway for 10 km.',
  'keep right and follow signs for Rashtrapathi Bhavan',
  'turn left onto Markham street and then turn right on to Himayat chowk',
  'In 300 m. your destination will be on the right',
  'On your way to Delhi public school traffic is lighter than usual',
  'The route via Faridabad bypass road is the fastest',
  'Head south east on State Bank road',
  'Head north west on Bank of Baroda road',
  'Mahatma gandhi marg',
  'Narendra Modi is our current Prime Minister',
  'Atal bihari vajpayee road',
  'Ramapuram is situated in southern coast of india',
  'Muthalakkodam is a popular tourist spot',
  'Panhartoo is a crowded place',
  'you have reached Gangappa Layout, 5th Cross',
  '50 kilometer further is Nanjundapuram',
  'Capt. Marshell is  patroling at 5th avenue',
  'mr. and mrs. smith',
  '$3.50 for gas',
  'Nemar jr. is a play actor',
  'Govindpuri extn. now has metro connectivity'
]


def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())
  synth = Synthesizer()
  synth.load(args.checkpoint)
  base_path = get_output_base_path(args.checkpoint)
  for i, text in enumerate(sentences):
    path = '%s-%d.wav' % (base_path, i)
    print('Synthesizing: %s' % path)
    with open(path, 'wb') as f:
      f.write(synth.synthesize(text))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
