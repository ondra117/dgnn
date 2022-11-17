from abc import ABC, abstractmethod

class Activation_fn(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, z):
        return self.fn(z)

    @abstractmethod
    def fn(self, z):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, z):
        raise NotImplementedError

    @abstractmethod
    def cuda_fn(self):
        raise NotImplementedError

    @abstractmethod
    def cuda_derivative(self):
        raise NotImplementedError

class Relu(Activation_fn):
    def __init__(self):
        """A rectified linear ctivation function."""
        super().__init__()

    def __str__(self):
        return "ReLU"

    def fn(self, z):
        return max(0, z)

    def derivative(self, z):
        return 0 if z < 0 else 1

    def cuda_fn(self):
        def fn(z):
            return max(0, z)
        return fn

    def cuda_derivative(self):
        def fn(z):
            return 0 if z < 0 else 1
        return fn

class Identity(Activation_fn):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Identity"

    def fn(self, z):
        return z

    def derivative(self, z):
        return 1

    def cuda_fn(self):
        def fn(z):
            return z
        return fn
    
    def cuda_derivative(self):
        def fn(z):
            return 1
        return fn