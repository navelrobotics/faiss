#!/usr/bin/python3
import faiss
import numpy

faiss.Kmeans(10, 20, verbose=True).train(numpy.random.rand(1000, 10).astype('float32'))
