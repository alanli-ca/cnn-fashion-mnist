import numpy as np
import uuid

def print_final_error(train_error, valid_error, test_error):
    print("final train error: {0:0.4f}".format(train_error))
    print("final valid error: {0:0.4f}".format(valid_error))
    print("final test error: {0:0.4f}".format(test_error))

def print_best_valid_epoch(train_errors, valid_errors, test_errors):
    best_valid_error_epoch = np.argmin(valid_errors)
    print("best valid error epoch: {}".format(best_valid_error_epoch))
    print("train error at epoch {0}: {1:0.4f}".format(best_valid_error_epoch, train_errors[best_valid_error_epoch]))
    print("valid error at epoch {0}: {1:0.4f}".format(best_valid_error_epoch, valid_errors[best_valid_error_epoch]))
    print("test error at epoch {0}: {1:0.4f}".format(best_valid_error_epoch, test_errors[best_valid_error_epoch]))

def write_errors_to_file(train_errors, valid_errors, test_errors, model_configs, model_name):
    best_valid_error_epoch = np.argmin(valid_errors)
    path = "results/"
    filename = "{}{:0.6f}_{}.txt".format(path, test_errors[best_valid_error_epoch], uuid.uuid4())
    train_errors_trunc = ["{0:0.6f}".format(i) for i in train_errors]
    valid_errors_trunc = ["{0:0.6f}".format(i) for i in valid_errors]
    test_errors_trunc = ["{0:0.6f}".format(i) for i in test_errors]
    model_configs = str(model_configs)
    with open(filename, 'w') as file_object:
        file_object.write("model_name:\n{}\n\n".format(model_name))
        file_object.write(str("train_errors:\n{}\n\n".format(train_errors_trunc)))
        file_object.write(str("valid_errors:\n{}\n\n".format(valid_errors_trunc)))
        file_object.write(str("test_errors:\n{}\n\n".format(test_errors_trunc)))
        file_object.write("model_configs:\n{}\n\n".format(model_configs))