import os

os.environ["NUMBA_ENABLE_CUDASIM"] = "0"

import numpy as np
from activation_fn import Relu, Identity
from numba import cuda
import math

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
        self.memory = np.zeros([1, 2, self.n_inputs + self.n_outputs], dtype=np.float64)
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

    def Calculate_CPU(self, inputs: np.ndarray, clear_mem: bool=False) -> np.ndarray:
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

    def Create_kernel(self):
        """
        Creates a kernel from computing beckprop and calculation, and moves network parameters to gpu
        """
        # mov all parameters to gpu
        self.cuda_weights = cuda.to_device(self.weights)
        self.cuda_neuron_info = cuda.to_device(self.neuron_info)
        self.cuda_memory = cuda.to_device(np.zeros_like(self.memory))

        # compile all functions to gpu
        hiden_fn = cuda.jit(self.hiden_activation.cuda_fn(), device=True)
        hiden_derivative = cuda.jit(self.hiden_activation.cuda_derivative(), device=True)
        last_fn = cuda.jit(self.last_activation.cuda_fn(), device=True)
        last_derivative = cuda.jit(self.last_activation.cuda_derivative(), device=True)

        @cuda.jit
        def kernel_calculate_one(weights_io, neuron_info_io, memory_io, inputs_io, outputs_io):
            for inpidx, inp in enumerate(inputs_io):
                for idx in range(math.ceil(inp.shape[0] / cuda.blockDim.x)):
                    idx_mem_mov = cuda.threadIdx.x + cuda.blockDim.x * idx
                    if inp.shape[0] > idx_mem_mov:
                        memory_io[0, 0, idx_mem_mov] = inp[idx_mem_mov]
                cuda.syncthreads()

                for layer_info in neuron_info_io:
                    if math.isnan(layer_info[cuda.threadIdx.x][2]): break

                    result = 0
                    for weight, mem in zip(weights_io[layer_info[cuda.threadIdx.x, 1]], memory_io[0, 0]):
                        if math.isnan(weight): weight = 0

                        result += weight * mem
                    result += layer_info[cuda.threadIdx.x, 0]
                    if layer_info[cuda.threadIdx.x, 3] == 1:
                        result = hiden_fn(result)
                    else:
                        result = last_fn(result)
                        outputs_io[inpidx, cuda.threadIdx.x] = result
                    

                    cuda.syncthreads()
                    memory_io[0, 0, layer_info[cuda.threadIdx.x, 2]] = result
                    cuda.syncthreads()

        def kernel_calculate_full():
            raise NotImplementedError

        def kernel_beckprop():
            raise NotImplementedError

        self._kernel_calculate_one = kernel_calculate_one
        self._kernel_calculate_full = kernel_calculate_full
        self._kernel_beckprop = kernel_beckprop

    def Calculate_GPU(self, inputs: np.ndarray, clear_mem: bool=False) -> np.ndarray:
        if clear_mem:
            self.cuda_memory = cuda.to_device(np.zeros_like(self.memory))
        cuda_inputs = cuda.to_device(inputs)
        cuda_outputs = cuda.to_device(np.zeros([inputs.shape[0], self.n_outputs]))

        self._kernel_calculate_one[1, self.max_layer_len](self.cuda_weights, self.cuda_neuron_info, self.cuda_memory, cuda_inputs, cuda_outputs)
        
        return cuda_outputs.copy_to_host()

    def Beckprop(self, training_data: list, clear_mem: bool=False) -> int:
        raise NotImplementedError

        

if __name__ == '__main__':
    from pprint import pprint

    np.random.seed(8)
    dgnn = DGNN(2, 4, Relu(), Identity())
    dgnn.Prepare_memory()
    dgnn.Create_kernel()
    out = dgnn.Calculate_GPU(np.array([
        [1, 2],
        [3, 4],
        [0, -1],
        [1, -2]
    ]))
    pprint(out)

    # dgnn.Claculate_slow(np.array([
    #     [[1, 2], [3, 4]],
    #     [[2, 1], [6, 5]],
    #     [[6, 6], [1, 1]]
    # ]))