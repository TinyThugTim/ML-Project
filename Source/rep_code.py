#!/usr/bin/env python
# coding: utf-8

# In[1]:


from qiskit import *
from qiskit.ignis.verification.topological_codes import RepetitionCode
from qiskit.ignis.verification.topological_codes import GraphDecoder
from qiskit.ignis.verification.topological_codes import lookuptable_decoding, postselection_decoding


# In[2]:


d = 9
T = 2
code = RepetitionCode(d,T)
code.code_qubit


# In[3]:


for log in ['0','1']:
    print('\n========= logical',log,'=========\n')
    print( code.circuit[log] )


# In[4]:


#Here the strings from right to left represent the outputs of the syndrome measurement rounds,4
#followed by the final measurement of the code qubits.
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

def get_noise(p_meas,p_gate):

    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    error_gate1 = depolarizing_error(p_gate, 9)
    error_gate2 = error_gate1.tensor(error_gate1)

    noise_model = NoiseModel()
#     noise_model.add_all_qubit_quantum_error(error_meas, "measure")
    noise_model.add_all_qubit_quantum_error(error_gate1, ["cx"])

    return noise_model


# In[ ]:


noise_model = get_noise(0.5, 0.5)


# In[ ]:


def get_syndrome(code,noise_model,shots=1014):

    circuits = code.get_circuit_list()
    

    job = execute( circuits, Aer.get_backend('qasm_simulator'),noise_model=noise_model, shots=shots )
    raw_results = {}
    for log in ['0','1']:
        raw_results[log] = job.result().get_counts(log)
    
    return code.process_results( raw_results )


# In[ ]:


get_syndrome(code, noise_model)


# In[ ]:


def get_results(code,noise_model,shots=1014):
    circuits = code.get_circuit_list()

    job = execute( circuits, Aer.get_backend('qasm_simulator'),noise_model=noise_model, shots=shots )
    raw_results = {}        
    for log in ['0','1']:
        raw_results[log] = job.result().get_counts(log)
    
    processed_results = code.process_results(raw_results)
    
    return processed_results


# In[ ]:


def get_syndrome_new(processed_results):

    final_results = [[],[],[],[],[]]
    for log in ['0','1','0']:
        for row in processed_results[log]:    
            #print (row)
            segments = row.split()            
            #print (segments)
            for i in range(len(segments)):
                final_results[i].append(segments[i])
    
    return final_results    


# In[ ]:


processed_results = get_results(code, noise_model)
get_results(code, noise_model)


# In[ ]:


# print(len(processed_results['0']))
# print(len(processed_results['1']))

results = get_syndrome_new(processed_results)
# print("Col1", results[0][50])
# print("Col2", results[1][50])
# print("Col3", results[2][50])
# print("Col4", results[3][50])
#print("Col5", results[4][50])
for item in results[3]:
    print(item)
    #10011100
    #1 0  10011001
#print(len(results[4]))


# In[ ]:


code = RepetitionCode(d,2)

results = get_syndrome(code,noise_model=noise_model,shots=8192)

dec = GraphDecoder(code)

logical_prob_match = dec.get_logical_prob(results)

for log in ['0','1']:
    print('d =',d,',log =',log)
    print('logical error probability for matching =',logical_prob_match[log])
    print('')
print('')

