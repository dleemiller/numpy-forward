#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>  // Include math library for sqrt, exp, etc.


typedef void (*ActivationFunction)(double*, int);
typedef void (*DimensionalActivationFunction)(double*, npy_intp*, npy_intp*, int, int);

static PyObject* apply_activation(PyObject *self, PyObject *args, ActivationFunction activation_func) {
    PyArrayObject *np_arr = NULL;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &np_arr)) {
        return NULL;
    }

    np_arr = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)np_arr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (np_arr == NULL) {
        return NULL;
    }

    double *data = (double*)PyArray_DATA(np_arr);
    int size = (int)PyArray_SIZE(np_arr);

    activation_func(data, size); // Call the specific activation function

    Py_INCREF(np_arr);
    return (PyObject*)np_arr;
}

static PyObject* apply_dimensional_activation(PyObject *self, PyObject *args, DimensionalActivationFunction activation_func) {
    PyArrayObject *np_arr = NULL;
    int dim;

    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &np_arr, &dim)) {
        return NULL;
    }

    np_arr = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)np_arr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (np_arr == NULL) {
        return NULL;
    }

    int ndim = PyArray_NDIM(np_arr);

    // Adjust for negative dimensions
    if (dim < 0) {
        dim += ndim;
    }

    // Check if the dimension is valid
    if (dim < 0 || dim >= ndim) {
        PyErr_SetString(PyExc_ValueError, "Invalid dimension");
        return NULL;
    }

    npy_intp* shape = PyArray_SHAPE(np_arr);
    npy_intp* strides = PyArray_STRIDES(np_arr);

    activation_func((double*)PyArray_DATA(np_arr), shape, strides, dim, ndim);

    Py_INCREF(np_arr);
    return (PyObject*)np_arr;
}


// GELU Activation Function (Approximate version)
static void gelu_activation(double* data, int size) {
    for (int i = 0; i < size; i++) {
        double x = data[i];
        double x_cubed = x * x * x;
        double tanh_arg = (sqrt(2 / M_PI) * (x + 0.044715 * x_cubed));
        data[i] = 0.5 * x * (1 + tanh(tanh_arg));
    }
}

// ReLU Activation Function
static void relu_activation(double* data, int size) {
    for (int i = 0; i < size; i++) {
        if (data[i] < 0) {
            data[i] = 0;
        }
    }
}

// Sigmoid Activation Function
static void sigmoid_activation(double* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = 1.0 / (1.0 + exp(-data[i]));
    }
}


// Swish Activation Function
static void swish_activation(double* data, int size) {
    for (int i = 0; i < size; i++) {
        double val = data[i];
        data[i] = val / (1.0 + exp(-val));
    }
}


// Utility function to compute softmax along a specified dimension
static void softmax_activation(double* data, npy_intp* shape, npy_intp* strides, int dim, int ndim) {
    npy_intp i, j;
    double max_val, sum_exp;

    // Iterate over each slice along the specified dimension
    for (i = 0; i < shape[dim]; i++) {
        max_val = -INFINITY;
        sum_exp = 0.0;

        // Find the maximum value for numerical stability
        for (j = 0; j < shape[ndim - 1]; j++) {
            double val = *(double *)((char *)data + i * strides[dim] + j * strides[ndim - 1]);
            if (val > max_val) {
                max_val = val;
            }
        }

        // Compute the sum of exponentials
        for (j = 0; j < shape[ndim - 1]; j++) {
            double val = exp(*(double *)((char *)data + i * strides[dim] + j * strides[ndim - 1]) - max_val);
            sum_exp += val;
            *(double *)((char *)data + i * strides[dim] + j * strides[ndim - 1]) = val;
        }

        // Normalize
        for (j = 0; j < shape[ndim - 1]; j++) {
            *(double *)((char *)data + i * strides[dim] + j * strides[ndim - 1]) /= sum_exp;
        }
    }
}

// Abstract GLU Activation Function
static void abstract_glu_activation(double* data, npy_intp* shape, npy_intp* strides, int dim, int ndim, ActivationFunction gate_activation) {
    npy_intp half_dim_size = shape[dim] / 2;
    for (npy_intp i = 0; i < half_dim_size; i++) {
        for (npy_intp j = 0; j < shape[ndim - 1]; j++) {
            double *a = (double *)((char *)data + i * strides[dim] + j * strides[ndim - 1]);
            double *b = (double *)((char *)data + (i + half_dim_size) * strides[dim] + j * strides[ndim - 1]);
            
            double activated_b = *b; // Assign the value of b to activated_b
            gate_activation(&activated_b, 1); // Apply the activation function to activated_b
            
            *a = *a * activated_b; // Apply the gated activation
        }
    }
}

// GLU Activation Function
static void glu_activation(double* data, npy_intp* shape, npy_intp* strides, int dim, int ndim) {
    abstract_glu_activation(data, shape, strides, dim, ndim, sigmoid_activation);
}

// SwiGLU Activation Function
static void swiglu_activation(double* data, npy_intp* shape, npy_intp* strides, int dim, int ndim) {
    abstract_glu_activation(data, shape, strides, dim, ndim, swish_activation);
}

// Tanh Activation Function
static void tanh_activation(double* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = tanh(data[i]);
    }
}


// EXPOSE TO PYTHON
static PyObject* py_tanh_activation(PyObject *self, PyObject *args) {
    return apply_activation(self, args, tanh_activation);
}

static PyObject* py_relu_activation(PyObject *self, PyObject *args) {
    return apply_activation(self, args, relu_activation);
}

static PyObject* py_gelu_activation(PyObject *self, PyObject *args) {
    return apply_activation(self, args, gelu_activation);
}

static PyObject* py_swish_activation(PyObject *self, PyObject *args) {
    return apply_activation(self, args, swish_activation);
}

static PyObject* py_sigmoid_activation(PyObject *self, PyObject *args) {
    return apply_activation(self, args, sigmoid_activation);
}

static PyObject* py_softmax_activation(PyObject *self, PyObject *args) {
    return apply_dimensional_activation(self, args, softmax_activation);
}

static PyObject* py_glu_activation(PyObject *self, PyObject *args) {
    return apply_dimensional_activation(self, args, glu_activation);
}

// Similarly for SwiGLU if it's implemented
static PyObject* py_swiglu_activation(PyObject *self, PyObject *args) {
    return apply_dimensional_activation(self, args, swiglu_activation);
}


// Method definitions
static PyMethodDef ActivationMethods[] = {
    {"tanh", py_tanh_activation, METH_VARARGS, "apply tanh activation function"},
    {"relu", py_relu_activation, METH_VARARGS, "apply relu activation function"},
    {"gelu", py_gelu_activation, METH_VARARGS, "apply GELU activation function"},
    {"sigmoid", py_sigmoid_activation, METH_VARARGS, "apply Sigmoid activation function"},
    {"softmax", py_softmax_activation, METH_VARARGS, "apply Softmax activation function"},
    {"swish", py_swish_activation, METH_VARARGS, "apply Swish activation function"},
    {"glu", py_glu_activation, METH_VARARGS, "apply GLU activation function"},
    {"swiglu", py_swiglu_activation, METH_VARARGS, "apply GLU activation function"},
    {NULL, NULL, 0, NULL}
};


// Module initialization
static struct PyModuleDef activations_module = {
    PyModuleDef_HEAD_INIT,
    "inference.activations",
    NULL,  // Module documentation
    -1,
    ActivationMethods  // Methods
};

PyMODINIT_FUNC PyInit_activations(void) {
    PyObject *module;
    module = PyModule_Create(&activations_module);
    if (!module) {
        return NULL;
    }
    import_array(); // Necessary for NumPy
    return module;
}

