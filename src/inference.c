#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

// Forward declaration for the initialization of the submodule
PyMODINIT_FUNC PyInit_activations(void);

static struct PyModuleDef inference_module = {
    PyModuleDef_HEAD_INIT,
    "inference",
    NULL, // Module documentation
    -1,
    NULL, // Methods
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_inference(void) {
    PyObject *m;
    m = PyModule_Create(&inference_module);
    if (!m) {
        return NULL;
    }

    import_array(); // Necessary for NumPy
    PyObject *activations_module = PyInit_activations();
    if (!activations_module) {
        return NULL;
    }

    // Add activations as a submodule
    PyModule_AddObject(m, "activations", activations_module);

    return m;
}

