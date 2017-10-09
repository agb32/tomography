
#include <omp.h>
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

#include "matcov_styc.h"


//Define the python "Tomo" object
typedef struct {
    PyObject_HEAD
    struct tomo_struct tomoStruct;
} Tomo;

static int Tomo_init(Tomo *self, PyObject *args, PyObject *kwds)
{
    //The tomo object __init__ function
    // printf("\nInitialising the covmat object in C\n");

    // declare all the things needed by the tomo struct
    long Nw;
    PyArrayObject *X, *Y;
    double obs, DiamTel;
    PyArrayObject *Nsubap;
    PyArrayObject *Nssp;
    PyArrayObject *GsAlt, *type, *alphaX, *alphaY;
    PyArrayObject *XPup, *YPup, *thetaML, *diamPup, *sspSize;
    long Nlayer;
    PyArrayObject *cn2, *h, *L0;
    int ncpu, part;

    printf("\nAbout to parse params...\n");

    if (! PyArg_ParseTuple(args, "lO!O!ddO!O!O!O!O!O!O!O!O!O!O!lO!O!O!ii", &Nw, &PyArray_Type, &X, &PyArray_Type, &Y, &DiamTel, &obs, &PyArray_Type, &Nsubap, &PyArray_Type, &Nssp, &PyArray_Type, &GsAlt, &PyArray_Type, &type, &PyArray_Type, &alphaX, &PyArray_Type, &alphaY, &PyArray_Type, &XPup, &PyArray_Type, &YPup, &PyArray_Type, &thetaML, &PyArray_Type, &diamPup, &PyArray_Type, &sspSize, &Nlayer, &PyArray_Type, &cn2, &PyArray_Type, &h, &PyArray_Type, &L0, &ncpu, &part)) return NULL;

    printf("\nParsed Python params\n");

    self->tomoStruct.Nw = Nw;
    self->tomoStruct.X = (double*) PyArray_DATA(PyArray_Cast(X, NPY_DOUBLE));
    self->tomoStruct.Y = (double*) PyArray_DATA(PyArray_Cast(Y, NPY_DOUBLE));
    self->tomoStruct.DiamTel = DiamTel;
    self->tomoStruct.obs = obs;
    self->tomoStruct.Nsubap = (long*) PyArray_DATA(PyArray_Cast(Nsubap, NPY_LONG));
    self->tomoStruct.Nssp = (long*) PyArray_DATA(PyArray_Cast(Nssp, NPY_LONG));
    self->tomoStruct.GsAlt = (double*) PyArray_DATA(PyArray_Cast(GsAlt, NPY_DOUBLE));
    self->tomoStruct.type = (int*) PyArray_DATA(PyArray_Cast(type, NPY_INT));
    self->tomoStruct.alphaX = (double*) PyArray_DATA(PyArray_Cast(alphaX, NPY_DOUBLE));
    self->tomoStruct.alphaY = (double*) PyArray_DATA(PyArray_Cast(alphaY, NPY_DOUBLE));
    self->tomoStruct.XPup = (double*) PyArray_DATA(PyArray_Cast(XPup, NPY_DOUBLE));
    self->tomoStruct.YPup = (double*) PyArray_DATA(PyArray_Cast(YPup, NPY_DOUBLE));
    self->tomoStruct.thetaML = (double*) PyArray_DATA(PyArray_Cast(thetaML, NPY_DOUBLE));
    self->tomoStruct.diamPup = (double*) PyArray_DATA(PyArray_Cast(diamPup, NPY_DOUBLE));
    self->tomoStruct.sspSize = (double*) PyArray_DATA(PyArray_Cast(sspSize, NPY_DOUBLE));
    self->tomoStruct.Nlayer = Nlayer;
    self->tomoStruct.cn2 = (double*) PyArray_DATA(PyArray_Cast(cn2, NPY_DOUBLE));
    self->tomoStruct.h = (double*) PyArray_DATA(PyArray_Cast(h, NPY_DOUBLE));
    self->tomoStruct.L0 = (double*) PyArray_DATA(PyArray_Cast(L0, NPY_DOUBLE));
    self->tomoStruct.ncpu = ncpu;
    self->tomoStruct.part = part;

    printf("loaded all data from %ld WFSs successfully\n", Nw);

    return 0;
}

static PyObject* covmat(Tomo *self, PyObject *args)
{
    printf("In the covmat function\n");
    PyArrayObject *covmat_npy;
    double *covmat_data;
    int ndim;
    npy_intp *dims;

    printf("Going to make a covmat!\n");
    if (! PyArg_ParseTuple(args, "O!", &PyArray_Type, &covmat_npy)) return NULL;

    printf("Parsed the covmat args\n");
    // Get a pointer to the data from the numpy array
    // if (PyArray_DTYPE(covmat_npy)->type_num != NPY_DOUBLE)
    // {
        printf("Cast covmat buffer to double...\n");
        printf("Current dtype: %d\n", PyArray_DTYPE(covmat_npy)->type_num);
        covmat_npy = PyArray_Cast(covmat_npy, NPY_DOUBLE);
    // }

    covmat_data = (double*) PyArray_DATA(covmat_npy);

    printf("Got the covmat array from numpy\n");
    //Do the calculation
   
    double begin = omp_get_wtime();
    matcov_styc(self->tomoStruct, covmat_data);
    double  end = omp_get_wtime();
    double d_time = (double)(end - begin);

    printf("Covmat execution time: %fs\n", d_time);


    // Send the pointer back to the python user, get params from input array
    ndim = PyArray_NDIM(covmat_npy);
    dims = PyArray_DIMS(covmat_npy);
    return Py_BuildValue("O", PyArray_SimpleNewFromData(ndim, dims, NPY_DOUBLE, covmat_data));
    //return Py_BuildValue("O", covmat_npy);
}

//Python wrapping
static void Tomo_dealloc(Tomo *self)
{
    self->ob_type->tp_free((PyObject*)self);
}

static PyMemberDef Tomo_members[] = {
        {NULL}
};

static PyMethodDef Tomo_methods[] = {
    {"covmat", (PyCFunction)covmat, METH_VARARGS, "Fills array with covariance matrix"},
    {NULL}
};

static PyTypeObject TomoType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "TomoAO.Tomo",    /*tp_name*/
    sizeof(Tomo), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)Tomo_dealloc,                         /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,        /*tp_flags*/
    "Wrapper around Erics covmat c code",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    Tomo_methods,             /* tp_methods */
    Tomo_members,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Tomo_init,      /* tp_init */
    0,                         /* tp_alloc */
    //Noddy_new,                 /* tp_new */
};

static PyMethodDef TomoAO_methods[] = {
	{NULL}
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC initTomoAO(void)
{
    PyObject* m;

	import_array();

    TomoType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&TomoType) < 0)
        return;

    m = Py_InitModule3("TomoAO", TomoAO_methods,
                       "A module for creating covariance matrices");

    Py_INCREF(&TomoType);
    PyModule_AddObject(m, "Tomo", (PyObject *)&TomoType);
}
