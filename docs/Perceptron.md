---
title: Program 1 - Perceptron
author: Matthew Krohn
date: 2020-09-25
pagestyle: empty
papersize: letter
geometry:
   - margin=0.5in
header-includes:
   - \usepackage{}
   - \hypersetup{colorlinks=true,
            allbordercolors={0 0 0},
            pdfborderstyle={/S/U/W 1}}
---

\pagenumbering{gobble}

# What Does a Perceptron Do?

To be honest, I'm still kind of fuzzy on all the math notation, but I'll do my
best to explain my understanding without it.

Let's say we have a bunch of data that describes different features of things we
wish to categorize into two sets. We as the trainer already know which pieces of
data belong to which set, and we can train the perceptron on two or more of
these features to split the data points into the two sets by a line. It's
visually almost like a linear regression, but perpendicular; we do not wish to
express a line through the data that unifies it, seeking instead to split the
data points into their two predefined groups. This can be extended into any
number of features and dimensions, but your error rate would probably go up for
most things.

Since we are training to categorize two different species, we trim our data set
to only have the selected species. Then we select how many features we want to
train on; the supplied plot function only works on two feature fits, but the
perceptron itself will fit with any number.

As the trainer, we know which species each piece of data belongs to, but the
perceptron does not. The perceptron takes the data and effectively "budges" a
see-saw back and forth along each point, using our supplied idea of what side of
the line the point should be on to readjust where it is placing the line between
all the data points. It can take a few iterations through all the data points,
but eventually the line gets "pushed" far enough in between the two
sets. This is not guaranteed to happen if the given data points are not linearly
separable, so it is important to not run forever trying to fix something that
cannot be fixed.

What is going on under the hood is some linear algebra, predicting, and
compensation for bad predictions. It's all very complicated and while I could
explain the weight vector and the bias and how a bad prediction affects the
weight vector at a very basic level, I implore you to exercise that well trained
Google-fu and have someone more qualified explain it. It's kinda magic.

## Requirements

Machine learning, even at the level of a lowly perceptron, is complicated
business; we're reinventing the wheel here, so we're gonna use some other
peoples' hard work in place of our sanity.

You'll need the following `python` libraries installed: `pandas`, `numpy`, and
`matplotlib`. While these are probably included in your \*nix distribution's
repositories (if you are so blessed), it is probably best to use a [virtual
environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
to install these so you don't muck things up for any other `python` projects you
may have that also happen to use the aforementioned libraries. If you insist on
installing them globally, the following incantations should suffice:


```sh
$ pip3 install pandas
$ pip3 install numpy
$ pip3 install matplotlib
```

## Usage

As this is meant to be a library, there is no CLI for this; the user is expected
to be able to make some minor edits to the supplied files to change what species
and features are being fit. To ease this as much as possible, one should only
need to edit the supplied `main.py` file (specifically the last 20 lines).

Once a pair of species and some features have been selected and put into the
call to `load_data`, running is simple:


```sh
$ python3 main.py
```

After a short delay, you should see a plot pop up that shows the selected
features on a scatter plot and the line that the perceptron made to categorize
them. A list with the number of errors in each iteration will be printed to the
console.

### Postscript

I apologize for my brevity in the preceding. I have no excuses to offer. I am
ashamed to admit that my deep understanding of the material thus far is not up
to par. I have been struggling with a lot, even disregarding school
responsibilities, and I am taking steps to fix it.
