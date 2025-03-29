import h5py

def inspect_model_keys(model_path):
    """ Inspect keys inside an HDF5 model file. """
    with h5py.File(model_path, "r") as f:
        print("Keys found in the model file:")
        for key in f.keys():
            print(key)

# Example usage:
inspect_model_keys("model.h5")