"""
Utility functions for setting up environment variables and configurations
to ensure smooth execution of deep learning code.
"""
import os
import platform
import logging

def setup_openmp_env():
    """
    Configure OpenMP environment variables to avoid conflicts between
    different libraries that use OpenMP (like PyTorch and NumPy).
    
    This addresses the "multiple copies of the OpenMP runtime" error.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Optional: Set number of threads if needed
    # Uncomment and adjust these as necessary for your hardware
    # os.environ["OMP_NUM_THREADS"] = "4"  
    # os.environ["MKL_NUM_THREADS"] = "4"

def setup_cuda_env():
    """
    Configure CUDA environment variables for better GPU performance.
    """
    # Allow TensorFlow/PyTorch to allocate GPU memory as needed
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    
    # Use TF32 precision on Ampere GPUs for better performance
    os.environ["NVIDIA_TF32_OVERRIDE"] = "1"

def configure_logging(level=logging.INFO):
    """
    Set up basic logging configuration.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def setup_environment(use_openmp=True, use_cuda=True, log_level=logging.INFO):
    """
    Main function to set up all environment configurations.
    
    Args:
        use_openmp: Whether to configure OpenMP settings
        use_cuda: Whether to configure CUDA settings
        log_level: Logging level
    """
    if use_openmp:
        setup_openmp_env()
    
    if use_cuda:
        setup_cuda_env()
    
    configure_logging(log_level)
    
    # Log system information
    logging.info(f"Environment setup complete. Platform: {platform.platform()}")

if __name__ == "__main__":
    # Example usage
    setup_environment()
    logging.info("Environment variables configured successfully")
