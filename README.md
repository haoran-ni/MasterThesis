# MasterThesis

## Abstract

The integration of dispersion interactions in atomic machine learning (ML) models has been a challenging topic. People have used methods such as baselining and local parametrization to approximate explicitly the dispersion behavior at long distances. The long-distance equivariant (LODE) framework was recently proposed as a data-driven ML method to learn the long-range interactions in atomic systems, but more examinations of the capability of this method are yet to be performed, especially how well it can capture the long-range interactions from a general data set. In this project, we will study and compare three different types of ML models, namely a pure short-range model using SOAP (Smooth Overlap of Atomic Positions) features, a short-range model using SOAP features combined with an explicit $r^{-6}$ (R6) model, a multiscale model using combined features of SOAP and LODE. All three types of models are trained and compared upon two data sets, the exfoliation of black phosphorus data set and the phosphorus allotropes data set. We will show explicitly that localized short-range models are not able to learn the dispersion interaction from the training set. In contrast, the multiscale model combining SOAP and LODE features is able to accurately capture the dispersion interaction from a general phosphorus data set, and can correctly reproduce the binding curve of black phosphorus.

## How to use it

This repository contains all codes needed to reproduce the results in my master thesis. To obtain the same results, the versions of the packages are given below.

- **equistore**: commit 5b6bc28b83540abdc28142e5d89f95d9c5a16b78 (HEAD -> master, origin/master, origin/HEAD)
- **librascal**: commit dcff1c3c06400a9ed83021ae9e9f61ca2d4a3ebb (HEAD -> phosphorus, origin/project/phosphorus)
- **rascaline**: commit 4f1a9baca89d2a9ab4b1ee5d03749c657fb3e861 (HEAD -> master, origin/master, origin/HEAD)
- **equistore-examples**: commit ccab6b5cf0d7506244e314dce5ac7aa26a56ea9d (HEAD -> feat/combined_models, origin/feat/combined_models)
- **pylode**: commit 982e2a72bcfd9d483c416269cdc1b5a317e2f8c4 (HEAD -> exponentFristVersion)
- **scikit-cosmo**: commit f7cf1d6185a6375029f21881aef2d2c1d492046c (HEAD -> main, origin/main, origin/HEAD)

The *utils* folder on top is copied from [equistore-examples](https://github.com/lab-cosmo/equistore-examples). It is put here for people who wants to run the codes. The phosphorus data sets come from [the exfoliation data set](https://github.com/libAtoms/testing-framework/tree/public/tests/P/black_exfoliation) and [the phosphorus allotropes data set](https://zenodo.org/record/4003703#.YuDv3XZBxEY).
