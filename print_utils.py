import numpy as np
import uuid

def print_final_error(train_error, valid_error, test_error):
    print("final traing error: {0:0.4f}".format(train_error))
    print("final valid error: {0:0.4f}".format(valid_error))
    print("final test error: {0:0.4f}".format(test_error))

def write_errors_to_file(train_errors, valid_errors, test_error, model_configs, model_name):
    path = "results/"
    filename = "{}{:0.6f}_{}.txt".format(path, test_error, uuid.uuid4())
    train_errors_trunc = ["{0:0.6f}".format(i) for i in train_errors]
    valid_errors_trunc = ["{0:0.6f}".format(i) for i in valid_errors]
    model_configs = str(model_configs)
    with open(filename, 'w') as file_object:
        file_object.write("model_name:\n{}\n\n".format(model_name))
        file_object.write(str("train_errors:\n{}\n\n".format(train_errors_trunc)))
        file_object.write(str("valid_errors:\n{}\n\n".format(valid_errors_trunc)))
        file_object.write("test_error:\n{:0.6f}\n\n".format(test_error))
        file_object.write("model_configs:\n{}\n\n".format(model_configs))