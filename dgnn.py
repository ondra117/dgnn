import numpy as np
from activation_fn import Relu, Identity

class DGNN:
    def __init__(self, n_inputs: int, n_outputs: int, hiden_activation, last_activation):
        """
        Directed Graph Neural Network

        Parameters
        ----------
        n_inputs:
            number of inputs
        n_outputs:
            number of outputs
        """
        #save all parameters
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hiden_activation = hiden_activation
        self.last_activation = last_activation

        self._inicialize()
    
    def _inicialize(self):
        self.weights = np.full([self.n_outputs, self.n_outputs + self.n_inputs], fill_value=None, dtype=np.float64)
        #fill weights randomly
        for i in range(self.n_inputs):
            self.weights.T[i] = np.random.normal(size=[self.weights.shape[0]])
        #there is saved bias, index of weights, index of memory, activation
        self.neuron_info = np.zeros([1, self.n_outputs, 4], dtype=np.int64)
        #fill index of memory
        self.neuron_info[0].T[2] = np.arange(self.n_inputs, self.n_inputs + self.n_outputs, 1, dtype=np.int64)
        #fill index of weights
        self.neuron_info[0].T[1] = np.arange(0, self.n_outputs, 1, dtype=np.int64)

        self.max_layer_len = self.n_outputs

    def Prepare_memory(self, n_sequences: int=1):
        """
        Prepare the memory for training or calculation.\n

        Parameters
        ----------
        n_sequences:
            Number of sequences\n
            Is intended for beckprop, it determines how many sequences beckprop can process at once.\n
            If you only want to perform the calculation, leave the value at 1.
        """

        self.memory = np.zeros([n_sequences, 2, self.n_inputs + self.n_outputs], dtype=np.float64)

    def Calculate_CPU(self, inputs: np.ndarray, clear_mem: bool=True) -> np.ndarray:
        if clear_mem:
            self.memory.fill(0)
        temporary_memory = np.copy(self.memory)
        output = np.zeros([inputs.shape[0], self.n_outputs])

        for idx, inp in enumerate(inputs):
            self.memory[0][0][:self.n_inputs] = inp
            
            for layer_info in self.neuron_info:
                for neuron_info in layer_info:
                    result = np.sum(np.nan_to_num(self.weights[neuron_info[1]]) * self.memory[0][0]) + neuron_info[0]
                    if neuron_info[3]:
                        result = self.hiden_activation(result)
                    else:
                        result = self.last_activation(result)
                    temporary_memory[0][0][neuron_info[2]] = result
                self.memory[0][0] = temporary_memory[0][0]
            
            output[idx] = self.memory[0][0][-self.n_outputs:]
        
        return output

    
        

        

if __name__ == '__main__':
    np.random.seed(8)
    dgnn = DGNN(2, 4, Relu(), Identity())
    dgnn.Prepare_memory()
    out = dgnn.Calculate_CPU(np.array([
        [1, 2],
        [3, 4],
        [0, -1],
        [1, 1]
    ]))
    print(out)
    # dgnn.Claculate_slow(np.array([
    #     [[1, 2], [3, 4]],
    #     [[2, 1], [6, 5]],
    #     [[6, 6], [1, 1]]
    # ]))