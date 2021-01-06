import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
#import kenlm
from math import log
from numpy import array
from numpy import argmax
import BestPath
from itertools import groupby

import models.crnn as crnn
import params
import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument('-m', '--model_path', type = str, required = True, help = 'crnn model path')
#parser.add_argument('-i', '--image_path', type = str, required = True, help = 'demo image path')
#args = parser.parse_args()

#model_path = 'netCRNN_4_5000.pth'
model_path = 'netCRNN_40_1000.pth'
#image_path = args.image_path

# net init
nclass = len(params.alphabet) + 1
new_classes = "-" + params.alphabet
use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cpu')

model = crnn.CRNN(params.imgH, params.nc, nclass, params.nh)
if torch.cuda.is_available():
    model = model.cuda()

# load model
print('loading pretrained model from %s' % model_path)
if params.multi_gpu:
    model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(model_path,map_location=DEVICE))

converter = utils.strLabelConverter(params.alphabet)

transformer = dataset.resizeNormalize((params.imgW, params.imgH))

#lm_model = kenlm.LanguageModel('telugu.binary')



def ctcBestPath_decode(seq, classes):
    "implements best path decoding as shown by Graves (Dissertation, p63)"

    # get char indices along best path
    #best_path = np.argmax(mat, axis=1)

    # collapse best path (using itertools.groupby), map to chars, join char list to string
    blank_idx = 0 #len(classes)
    best_chars_collapsed = [classes[k] for k, _ in groupby(seq) if k != blank_idx]
    res = ''.join(best_chars_collapsed)
    return res

def calc_lm_score(sequence):
    tel_string = ctcBestPath_decode(sequence,new_classes)
    lm_score = lm_model.score(tel_string)
    return lm_score


# beam search
def beam_search_decoder_with_lm(data, k):
	sequences = [[list(), 0.0]]
	# walk over each step in sequence
	for row in data:
		all_candidates = list()
		# expand each current candidate
		for i in range(len(sequences)):
			seq, score = sequences[i]
			for j in range(len(row)):
				candidate = [seq + [j], score - row[j]-((0.01)/calc_lm_score(seq + [j]))]
				all_candidates.append(candidate)
		# order all candidates by score
		ordered = sorted(all_candidates, key=lambda tup:tup[1])
		# select k best
		sequences = ordered[:k]
	return sequences


def predict_word(img):
    #image = Image.open(img).convert('L')
    image = transformer(img)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)
    model.eval()
    preds = model(image)
    ctc_in_matrix = preds.transpose(1, 0)
    ctc_in_matrix = ctc_in_matrix.squeeze(0)
    ctc_matrix_numpy = ctc_in_matrix.detach().cpu().numpy()
    #seq = beam_search_decoder_with_lm(ctc_matrix_numpy,5)
    #result_beam = ctcBestPath_decode(seq[0][0],new_classes)
    

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.LongTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))

    return_result = sim_pred #result_beam


    return return_result
