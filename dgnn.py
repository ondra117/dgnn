import numpy as np
from activation_fn import Relu, Identity
from numba import cuda

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
        self.cuda_weights = cuda.to_device(np.zeros_like(self.weights))
        self.cuda_neuron_info = cuda.to_device(self.neuron_info)
        self.cuda_memory = cuda.to_device(self.memory)

        # compile all functions to gpu
        hiden_fn = cuda.jit(self.hiden_activation.cuda_fn(), device=True)
        hiden_derivative = cuda.jit(self.hiden_activation.cuda_derivative(), device=True)
        last_fn = cuda.jit(self.last_activation.cuda_fn(), device=True)
        last_derivative = cuda.jit(self.last_activation.cuda_derivative(), device=True)

        @cuda.jit
        def kernel_calculate_one(weights_io, neuron_info_io, memory_io, inputs_io):
            for inp in inputs_io:
                if inp.shape[0] > cuda.threadIdx.x:
                    memory_io[0, 0, cuda.threadIdx.x] = inp[cuda.threadIdx.x]
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

        self._kernel_calculate_one[1, self.max_layer_len](self.cuda_weights, self.cuda_neuron_info, self.cuda_memory, cuda_inputs)
        return self.cuda_memory.copy_to_host()

        

if __name__ == '__main__':
    np.random.seed(8)
    dgnn = DGNN(2, 4, Relu(), Identity())
    dgnn.Prepare_memory()
    dgnn.Create_kernel()
    out = dgnn.Calculate_GPU(np.array([
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