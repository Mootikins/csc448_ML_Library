# Machine Learning: My (Not so) Big Book

This repo serves as a portfolio of sorts of some machine learning tools.

While my initial intention was to make all but the perceptron in Rust, a few
hours of fiddling led me to to conclude what the past half-decade of machine
learning engineers have already decided: Python is just way better at it. So I
gave in against my general dislike for Python (sue me) and just wrote the stuff.
I will say though, it was a lot easier than I think it would have been in
basically any other language. Thanks `numpy`.

#### Advisement about LaTeX

I use a lot of [**Pandoc**](https://pandoc.org/), so the more mathematical sections
may include LaTeX. I will generally try my best to use fairly simple TeX
notation when possible, but if you have trouble reading it or would prefer a
PDF, try your hand at compiling this document with it:

```sh
pandoc -f markdown README.md -o README.pdf
```

## What's in here?

This repo contains a Python library in the [ML](ML) folder which can be imported
like so:

```py
from ML.Perceptron import Perceptron
```

I am not a snake charmer, so I will apologize in advance for possibly not
following best or common practices.

There is also a K nearest neighbors implementation written in C in the
[corresponding submodule](c-omp-k_nn/).

## The Library Stuff

The library contains the following:

* [X] [Perceptron](src/ML/Perceptron.py) 
   * [Example](src/perceptron_example.py)
   * [Documentation and Theory](docs/Perceptron.md)
* [X] [Linear Regression](src/ML/LinearRegression.py)
   * [Example](src/linear_regression_example.py)
   * [Documentation and Theory](docs/LinearRegression.md)
* [ ] [Decision Stumps](src/ML/DecisionStump.py)
   * [Example](src/decision_stump_example.py)
   * [Documentation and Theory](docs/DecisionStumps.md)
* [ ] [Soft Support Vector Machine](src/ML/SoftSVM.py)
   * [Example](src/soft_svm_example.py)
   * [Documentation and Theory](docs/SoftSVM.md)
* [ ] [Logistic Regression](src/ML/LogisticRegression.py)
   * [Example](src/logistic_regression_example.py)
   * [Documentation and Theory](docs/LogisticRegression.md)
* [ ] [K Nearest Neighbors](c-omp-k_nn/)
   * [Example](https://github.com/mootikins/c-omp-k_nn)
   * [Documentation and Theory](docs/K-NN.md)

While the *library* only requires `numpy` and `pandas`, the examples also use
`matplotlib`, so it is recommended you use a virtual environment or other method
of your choice to make sure they are available. It should also go without saying
that you need python 3 installed. What exact version, I am unsure, but I've been
using 3.8 and 3.9.
