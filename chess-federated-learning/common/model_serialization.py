"""
Framework-agnostic model serialization for federated learning.

This module provides unified serialization interfaces for different deep learning
frameworks (PyTorch, TensorFlow), allowing the protocol layer to remain
framework-independent.

Key Features:
    - Abstract serializer interface for framework independence
    - PyTorch state_dict serialization
    - TensorFlow weights serialization
    - Optional compression for bandwidth optimization
    - Multiple encoding formats (binary, base64)

Usage:
    # For PyTorch
    serializer = PyTorchSerializer(compression=True)
    data = serializer.serialize(model.state_dict())
    
    # For TensorFlow
    serializer = TensorFlowSerializer(compression=True)
    data = serializer.serialize(model.get_weights())
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
import io
import gzip
import base64
from loguru import logger


class ModelSerializer(ABC):
    """
    Abstract base class for model serializers.
    
    This interface ensures all framework-specific serializers provide
    the same methods, allowing the rest of the system to be framework-agnostic.
    
    Attributes:
        compression: Whether to compress serialized data with gzip
        encoding: Encoding format ('binary' or 'base64')
    """
    
    def __init__(self, compression: bool = False, encoding: str = 'binary'):
        """
        Initialize the serializer.
        
        Args:
            compression: If True, compress data with gzip (reduces size by ~50-70%)
            encoding: 'binary' for raw bytes, 'base64' for JSON-compatible string
        """
        self.compression = compression
        self.encoding = encoding
        
        log = logger.bind(context="ModelSerializer.__init__")
        log.info(f"Initialized {self.__class__.__name__} with compression={compression}, encoding={encoding}")
    
    @abstractmethod
    def serialize(self, model_state: Any) -> Union[bytes, str]:
        """
        Serialize model state to transmittable format.
        
        Args:
            model_state: Framework-specific model state (state_dict, weights, etc.)
        
        Returns:
            Serialized data (bytes or base64 string depending on encoding)
        """
        pass
    
    @abstractmethod
    def deserialize(self, data: Union[bytes, str]) -> Any:
        """
        Deserialize data back to framework-specific model state.
        
        Args:
            data: Serialized data (bytes or base64 string)
        
        Returns:
            Framework-specific model state ready to load
        """
        pass
    
    def _compress(self, data: bytes) -> bytes:
        """
        Compress data using gzip.
        
        Args:
            data: Raw bytes to compress
        
        Returns:
            Compressed bytes
        """
        log = logger.bind(context=f"{self.__class__.__name__}._compress")
        original_size = len(data)
        compressed = gzip.compress(data, compresslevel=6)  # Level 6 = good balance
        compressed_size = len(compressed)
        ratio = (1 - compressed_size / original_size) * 100
        
        log.debug(f"Compressed {original_size} -> {compressed_size} bytes ({ratio:.1f}% reduction)")
        return compressed
    
    def _decompress(self, data: bytes) -> bytes:
        """
        Decompress gzip data.
        
        Args:
            data: Compressed bytes
        
        Returns:
            Decompressed bytes
        """
        log = logger.bind(context=f"{self.__class__.__name__}._decompress")
        compressed_size = len(data)
        decompressed = gzip.decompress(data)
        log.debug(f"Decompressed {compressed_size} -> {len(decompressed)} bytes")
        return decompressed
    
    def _encode(self, data: bytes) -> Union[bytes, str]:
        """
        Encode bytes according to specified encoding format.
        
        Args:
            data: Raw bytes
        
        Returns:
            Encoded data (unchanged bytes or base64 string)
        """
        log = logger.bind(context=f"{self.__class__.__name__}._encode")
        
        if self.encoding == 'base64':
            encoded = base64.b64encode(data).decode('utf-8')
            log.trace(f"Base64 encoded to string of length {len(encoded)}")
            return encoded
        else:
            log.trace(f"Returning raw binary of length {len(data)}")
            return data
    
    def _decode(self, data: Union[bytes, str]) -> bytes:
        """
        Decode data from encoding format to raw bytes.
        
        Args:
            data: Encoded data (bytes or base64 string)
        
        Returns:
            Raw bytes
        """
        log = logger.bind(context=f"{self.__class__.__name__}._decode")
        
        if isinstance(data, str):
            # Assume base64 encoded string
            decoded = base64.b64decode(data)
            log.trace(f"Base64 decoded string to {len(decoded)} bytes")
            return decoded
        else:
            log.trace(f"Data already in binary format ({len(data)} bytes)")
            return data


class PyTorchSerializer(ModelSerializer):
    """
    Serializer for PyTorch models.
    
    Handles PyTorch state_dict serialization using torch.save and torch.load.
    State dict is a dictionary mapping layer names to tensors:
        {'layer1.weight': tensor(...), 'layer1.bias': tensor(...), ...}
    
    Example:
        >>> import torch
        >>> model = MyModel()
        >>> serializer = PyTorchSerializer(compression=True)
        >>> data = serializer.serialize(model.state_dict())
        >>> state_dict = serializer.deserialize(data)
        >>> model.load_state_dict(state_dict)
    """
    
    def serialize(self, model_state: Dict) -> Union[bytes, str]:
        """
        Serialize PyTorch state_dict to bytes or base64 string.
        
        Process:
        1. Use torch.save to convert state_dict to bytes
        2. Optionally compress with gzip
        3. Optionally encode to base64 for JSON compatibility
        
        Args:
            model_state: PyTorch state_dict (dict of layer_name -> torch.Tensor)
        
        Returns:
            Serialized data ready for network transmission
        """
        log = logger.bind(context="PyTorchSerializer.serialize")
        
        try:
            import torch
        except ImportError:
            log.error("PyTorch not installed! Install with: pip install torch")
            raise ImportError("PyTorch is required for PyTorchSerializer")
        
        log.info(f"Serializing PyTorch model with {len(model_state)} layers")
        
        # Step 1: Convert state_dict to bytes using torch.save
        buffer = io.BytesIO()
        torch.save(model_state, buffer)
        data = buffer.getvalue()
        log.debug(f"PyTorch state_dict serialized to {len(data)} bytes")
        
        # Step 2: Optionally compress
        if self.compression:
            data = self._compress(data)
        
        # Step 3: Encode according to format
        encoded = self._encode(data)
        
        log.info(f"Serialization complete, final size: {len(encoded)} {'chars' if isinstance(encoded, str) else 'bytes'}")
        return encoded
    
    def deserialize(self, data: Union[bytes, str]) -> Dict:
        """
        Deserialize data back to PyTorch state_dict.
        
        Process:
        1. Decode from base64 if needed
        2. Decompress if needed
        3. Use torch.load to reconstruct state_dict
        
        Args:
            data: Serialized data (bytes or base64 string)
        
        Returns:
            PyTorch state_dict ready for model.load_state_dict()
        """
        log = logger.bind(context="PyTorchSerializer.deserialize")
        
        try:
            import torch
        except ImportError:
            log.error("PyTorch not installed! Install with: pip install torch")
            raise ImportError("PyTorch is required for PyTorchSerializer")
        
        log.info(f"Deserializing PyTorch model from {len(data)} {'chars' if isinstance(data, str) else 'bytes'}")
        
        # Step 1: Decode from encoding format
        decoded = self._decode(data)
        
        # Step 2: Decompress if needed
        if self.compression:
            decoded = self._decompress(decoded)
        
        # Step 3: Reconstruct state_dict using torch.load
        buffer = io.BytesIO(decoded)
        state_dict = torch.load(buffer, map_location='cpu')  # Load to CPU by default
        
        log.info(f"Deserialization complete, recovered {len(state_dict)} layers")
        log.debug(f"Layer names: {list(state_dict.keys())[:5]}...")  # Log first 5 layer names
        
        return state_dict


class TensorFlowSerializer(ModelSerializer):
    """
    Serializer for TensorFlow/Keras models.
    
    Handles TensorFlow model weights (list of numpy arrays) serialization.
    TensorFlow weights are typically obtained via model.get_weights():
        [array(...), array(...), ...]
    
    Example:
        >>> import tensorflow as tf
        >>> model = tf.keras.Model(...)
        >>> serializer = TensorFlowSerializer(compression=True)
        >>> data = serializer.serialize(model.get_weights())
        >>> weights = serializer.deserialize(data)
        >>> model.set_weights(weights)
    """
    
    def serialize(self, model_weights: List) -> Union[bytes, str]:
        """
        Serialize TensorFlow weights to bytes or base64 string.
        
        Process:
        1. Convert numpy arrays to lists (JSON-serializable)
        2. Use pickle to serialize the list
        3. Optionally compress with gzip
        4. Optionally encode to base64
        
        Args:
            model_weights: List of numpy arrays from model.get_weights()
        
        Returns:
            Serialized data ready for network transmission
        """
        log = logger.bind(context="TensorFlowSerializer.serialize")
        
        try:
            import numpy as np
            import pickle
        except ImportError:
            log.error("NumPy not installed! Install with: pip install numpy")
            raise ImportError("NumPy is required for TensorFlowSerializer")
        
        log.info(f"Serializing TensorFlow model with {len(model_weights)} weight arrays")
        
        # Step 1: Convert numpy arrays to serializable format
        # We convert to lists to ensure compatibility, then pickle
        serializable_weights = [w.tolist() if hasattr(w, 'tolist') else w for w in model_weights]
        
        # Log some statistics
        total_params = sum(np.array(w).size for w in serializable_weights)
        log.debug(f"Total parameters: {total_params:,}")
        
        # Step 2: Pickle the weights
        data = pickle.dumps(serializable_weights, protocol=pickle.HIGHEST_PROTOCOL)
        log.debug(f"TensorFlow weights pickled to {len(data)} bytes")
        
        # Step 3: Optionally compress
        if self.compression:
            data = self._compress(data)
        
        # Step 4: Encode according to format
        encoded = self._encode(data)
        
        log.info(f"Serialization complete, final size: {len(encoded)} {'chars' if isinstance(encoded, str) else 'bytes'}")
        return encoded
    
    def deserialize(self, data: Union[bytes, str]) -> List:
        """
        Deserialize data back to TensorFlow weights (numpy arrays).
        
        Process:
        1. Decode from base64 if needed
        2. Decompress if needed
        3. Unpickle to get list of arrays
        4. Convert back to numpy arrays
        
        Args:
            data: Serialized data (bytes or base64 string)
        
        Returns:
            List of numpy arrays ready for model.set_weights()
        """
        log = logger.bind(context="TensorFlowSerializer.deserialize")
        
        try:
            import numpy as np
            import pickle
        except ImportError:
            log.error("NumPy not installed! Install with: pip install numpy")
            raise ImportError("NumPy is required for TensorFlowSerializer")
        
        log.info(f"Deserializing TensorFlow model from {len(data)} {'chars' if isinstance(data, str) else 'bytes'}")
        
        # Step 1: Decode from encoding format
        decoded = self._decode(data)
        
        # Step 2: Decompress if needed
        if self.compression:
            decoded = self._decompress(decoded)
        
        # Step 3: Unpickle the weights
        weights_lists = pickle.loads(decoded)
        
        # Step 4: Convert back to numpy arrays
        weights = [np.array(w) for w in weights_lists]
        
        log.info(f"Deserialization complete, recovered {len(weights)} weight arrays")
        total_params = sum(w.size for w in weights)
        log.debug(f"Total parameters: {total_params:,}")
        
        return weights


def get_serializer(framework: str, **kwargs) -> ModelSerializer:
    """
    Factory function to get the appropriate serializer for a framework.
    
    This is the recommended way to obtain a serializer, as it abstracts
    away the specific serializer class.
    
    Args:
        framework: 'pytorch' or 'tensorflow'
        **kwargs: Additional arguments passed to serializer (compression, encoding)
    
    Returns:
        Appropriate ModelSerializer instance
    
    Raises:
        ValueError: If framework is not supported
    
    Example:
        >>> serializer = get_serializer('pytorch', compression=True)
        >>> data = serializer.serialize(model.state_dict())
    """
    log = logger.bind(context="get_serializer")
    log.info(f"Getting serializer for framework: {framework}")
    
    framework = framework.lower()
    
    if framework in ['pytorch', 'torch']:
        return PyTorchSerializer(**kwargs)
    elif framework in ['tensorflow', 'tf', 'keras']:
        return TensorFlowSerializer(**kwargs)
    else:
        log.error(f"Unsupported framework: {framework}")
        raise ValueError(f"Unsupported framework: {framework}. Use 'pytorch' or 'tensorflow'")