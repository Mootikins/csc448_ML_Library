# *K* Nearest Neighbors

*K* nearest neighbors is used for classification and regression of single- or
multi-variate data. This implementation reads a provided CSV file before reading
queries from `stdin` or a pipe, printing the classification or predicted result
value to `stdout`.

## Building, Running, Use

While these are included in the main project's README, I will briefly gloss over
them here.

```sh
# To build the debug and release targets
make debug
make knn

# To run debug release release
./debug [FLAGS and OPTIONS] FILE
```

The intention for this was to be relatively `*nix`-y, so it only reads CSVs,
ignores headers, and somewhat reliably auto-detects which columns is the
classification.

## Theory

K nearest neighbors is, thankfully, easy as hell. Read in your data points, then
your query (or queries). For each query, you compute its distance from each
data point (there are a variety of methods here, but I have only included
euclidean distance). You sort those distances and take the *k* shortest
distances and return either the mode of their classification or the mean of
their "label" values.

Doing this means that a point gets categorized  near its neighbors, making it
most useful for clustering problems where you are classifying the data as one of
a number of groups.
