#!/usr/bin/env python
# coding: utf-8

# In[1]:


from qiskit import *
from qiskit.ignis.verification.topological_codes import RepetitionCode
from qiskit.ignis.verification.topological_codes import GraphDecoder
from qiskit.ignis.verification.topological_codes import lookuptable_decoding, postselection_decoding
import numpy as np
import os, sys


# In[2]:


d = 3
T = 3
code = RepetitionCode(d,T)
code.code_qubit


# In[3]:


# for log in ['0','1']:
#     print('\n========= logical',log,'=========\n')
#     print( code.circuit[log] )


# In[4]:


circuits = code.get_circuit_list()
job = execute( circuits, Aer.get_backend('qasm_simulator') )
raw_results = {}
for log in ['0','1']:
    raw_results[log] = job.result().get_counts(log)
    print('\n========= logical',log,'=========\n')
    print(raw_results[log])


# In[5]:


code.process_results( raw_results )


# In[6]:


from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error

def get_noise(p_error,p_gate):
    bit_flip = pauli_error([('X', p_error), ('I', 1 - p_error)])
    phase_flip = pauli_error([('Z', p_error), ('I', 1 - p_error)])
    phase_flip2 = pauli_error([('Y', p_error), ('I', 1 - p_error)])
    bitphase_flip = bit_flip.compose(phase_flip)
    bitphase_flip2 = bitphase_flip.compose(phase_flip2)
    error_gate2 = bitphase_flip.tensor(bitphase_flip)

    noise_model = NoiseModel()
#     noise_model.add_all_qubit_quantum_error(error_meas, "measure")
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"])

    return noise_model


# In[7]:


noise_model = get_noise(0.5, 1)


# In[8]:


def get_syndrome(code,noise_model,shots=1014):

    circuits = code.get_circuit_list()


    job = execute( circuits, Aer.get_backend('qasm_simulator'),noise_model=noise_model, shots=shots )
    raw_results = {}
    for log in ['0','1']:
        raw_results[log] = job.result().get_counts(log)

    return code.process_results( raw_results )


# In[9]:


get_syndrome(code, noise_model)


# In[10]:


def get_results(code,noise_model,shots=1014):
    circuits = code.get_circuit_list()

    job = execute( circuits, Aer.get_backend('qasm_simulator'),noise_model=noise_model, shots=shots )
    raw_results = {}
    for log in ['0','1']:
        raw_results[log] = job.result().get_counts(log)

    processed_results = code.process_results(raw_results)

    return processed_results


# In[11]:


def get_syndrome_new(processed_results):

    final_results = [[],[],[],[],[],[]]
    for log in ['0','1']:
        for row in processed_results[log]:
            #print (row)
            segments = row.split()
            #print (segments)
            for i in range(len(segments)):
                final_results[i].append(segments[i])

    return final_results


# In[12]:


processed_results = get_results(code, noise_model)


# In[13]:


# print(len(processed_results['0']))
# print(len(processed_results['1']))

results = np.array(get_syndrome_new(processed_results)).T
results = results.tolist()

for row in range(len(np.array(results)[:, 0])):
    merged = ''.join(results[row][2:6])
    results[row][2:6] = merged

results = np.array(results)
print(results.shape)
print(results[0, :])


with open('../Datasets/d=3_test.txt', "w") as outfile:
    np.savetxt(outfile, results, fmt='%s')


print("Col1 \n", results[:49, 0])
print("Col2 \n", results[:49, 1])
print("Col3 \n", results[:49, 2])
print("Col4", results[:49, 3])
print("Col5", results[:49, 4])
print("Col6", results[:49, 5])
#print(results[3])
    #10011100
    #1 0  10011001
#print(len(results[4]))


# In[14]:


code = RepetitionCode(d,2)

results = get_syndrome(code,noise_model=noise_model,shots=8192)

dec = GraphDecoder(code)

logical_prob_match = dec.get_logical_prob(results)

for log in ['0','1']:
    print('d =',d,',log =',log)
    print('logical error probability for matching =',logical_prob_match[log])
    print('')
print('')
