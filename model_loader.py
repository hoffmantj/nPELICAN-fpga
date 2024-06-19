import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    nargs=1
)#path to PELICAN-nano model .pt file
args = parser.parse_args()

m = torch.load(args.model[0], map_location='cpu')
np.set_printoptions(precision=15, floatmode='fixed')
torch.set_printoptions(precision=15)

#calculate first batchnorm array
mean1 = m['model_state']['net2to2.message_layers.0.normlayer.running_mean'].item()
weight1 = m['model_state']['net2to2.message_layers.0.normlayer.weight'].item()
var1 = m['model_state']['net2to2.message_layers.0.normlayer.running_var'].item()
bias1 = m['model_state']['net2to2.message_layers.0.normlayer.bias'].item()
batch1= np.array((mean1, weight1/np.sqrt(var1), bias1))

#calculate second batchnorm array
mean2 = m['model_state']['msg_2to0.normlayer.running_mean'].numpy()
weight2 = m['model_state']['msg_2to0.normlayer.weight'].numpy()
var2 = m['model_state']['msg_2to0.normlayer.running_var'].numpy()
bias2 = m['model_state']['msg_2to0.normlayer.bias'].numpy()
batch2 = np.column_stack((mean2, weight2/np.sqrt(var2), bias2))


#write file
f = open("weights/weights.h","w")

f.write('#include "../nPELICAN.h"\n')
f.write('//model: ' + m['args'].prefix + '\n')
f.write('//nobj: ' + str(m['args'].nobj) + '\n\n')

#calculate normalization constants
f.write('//normalization constants\n')
f.write('//nobj avg = {}\n'.format(m['args'].nobj_avg))
f.write('internal_t invnave = {};\n'.format(1/m['args'].nobj_avg))
f.write('internal_t invnave2 = {};\n\n'.format(1/(m['args'].nobj_avg)**2))


f.write('//first batchnorm [mean, weight/sqrt(var), bias]\n')
f.write('weight_t batch1_2to2[3] = ' + np.array2string(batch1, separator=', ').replace('\n', '').replace('[', '{').replace(']', '}') + ';\n\n')


f.write('//2to2 linear layer\n')
f.write('weight_t w1_2to2[NHIDDEN*6] = ' + np.array2string(np.ravel(m['model_state']['net2to2.eq_layers.0.coefs'].numpy()[0]), separator=', ').replace('\n', '').replace('[', '{').replace(']', '}') + ';\n')
f.write('bias_t b1_2to2[NHIDDEN] = ' + np.array2string(m['model_state']['net2to2.eq_layers.0.bias'].numpy(), separator=', ').replace('\n', '').replace('[', '{').replace(']', '}') + ';\n')
f.write('bias_t b1_diag_2to2[NHIDDEN] = ' + np.array2string(m['model_state']['net2to2.eq_layers.0.diag_bias'].numpy(), separator=', ').replace('\n', '').replace('[', '{').replace(']', '}') + ';\n\n')


f.write('//second batchnorm [channel][mean, weight/sqrt(var), bias]\n')
f.write('weight_t batch2_2to0[NHIDDEN][3] = ' + np.array2string(batch2, separator=', ').replace('\n', '').replace('[', '{').replace(']', '}') + ';\n\n')


f.write('//2to1 linear layer\n')
f.write('weight_t w2_2to0[NHIDDEN*2*NOUT] = ' + np.array2string(np.ravel(m['model_state']['agg_2to0.coefs'].numpy()), separator=', ').replace('\n', '').replace('[', '{').replace(']', '}') + ';\n')
f.write('bias_t b2_2to0[NOUT] = ' + np.array2string(m['model_state']['agg_2to0.bias'].numpy()[0], separator=', ').replace('\n', '').replace('[', '{').replace(']', '}') + ';\n')


f.close()