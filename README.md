# General

## Requirements

Twarc-count requires Python 3.7 or greater and pip.

## Installation

You need to clone this repository.

`git clone https://github.com/DataPolitik/rt_changes.git`

And then, move to the folder **rt_changes**. Then, install all modules required by the script:

`pip install -r requirements.txt`

## Example of use

[Crowdosourced elite during the first wave of Covid19 in Spain](https://datapolitik.medium.com/el-baile-de-las-%C3%A9lites-en-twitter-9a288fb32eb3)

################################################################################

# RT_changes

Computes a ranking of authors who receive retweets on specific time intervals.

## Usage

`changes.py <INFILE> <OUTFILE> [-f [FIELDS] ]`

* **-g** | **- -granularity**: The time interval. You can use any offset alias for Pandas time series.
* **-a** | **- -alpha**: An inertia parameter that weighs retweets received in the previous time intervals (default = 0.005).
* **-t** | **- -threshold**: Removes users whose sum of scores are below the specific threshold.
* **-i** | **- -interval**: Specify a date period to process.
 
## Interval parameter

The paramenter -i waits for two dates separated by a comma (eg: start_time,end_time) the format should be according
YYYY-MM-DD-HH:MM:SS.

## Granularity

Some allowed values are:

* H: Hours
* M: Minutes
* Y: Years
* W: Weeks
* S: Seconds

A complete description of allowed aliases can be found at: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

## Examples

### Computes a simple ranking

`changes.py examples\results.json output.csv`

### Computes a weekly ranking

`changes.py examples\results.json output.csv -g W`

### Removes all user under 50 points

`changes.py examples\results.json output.csv -t 50`

### Compute data from an specific date interval

`changes.py  examples/results.json output -i 2021-10-18,2022-10-18`

###########################################################################

# Dendrogram.py

Computes distances between elites.
Calls clustering.py. Which return linkage matrix and polarisation.
Plots dendrogram.

## Usage

`dendrogram.py <INFILE1> <INFILE2> [-f [FIELDS] ]`

INFILE1 same as INFILE in changes.py
INFILE2 same as OUPTUT in changes.py

* **-g** | **- -granularity**: The time interval. You can use any offset alias for Pandas time series.
* **-a** | **- -alpha**: An inertia parameter that weighs retweets received in the previous time intervals (default = 0.005).
* **-t** | **- -threshold**: Removes users whose sum of scores are below the specific threshold.
* **-i** | **- -interval**: Specify a date period to process.
* **-m** | **- -method**: Specify clustering method used.
* **-l** | **- -algorithm**: Specify clustering algorithm used.
 
## Interval parameter

The paramenter -i waits for two dates separated by a comma (eg: start_time,end_time) the format should be according
YYYY-MM-DD-HH:MM:SS.

## Granularity

Some allowed values are:

* H: Hours
* M: Minutes
* Y: Years
* W: Weeks
* S: Seconds

A complete description of allowed aliases can be found at: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

## Algorithm

Allowed values are:

* generic: faster, but only works with method "ward"
* nn_chain: slower, but works with any distance update scheme

Clustering performed only if algorithm values are one of the above

## Method

Allowed values are:

* ward: works with any algorithm
* centroid: does not work if algorithm is "nn_chain"
* poldist: does not work if algorithm is "nn_chain"

## Examples

### Computes a simple ranking

`dendrogram.py examples\results.json examples\elites.csv

### Computes a weekly ranking

`dendrogram.py examples\results.json examples\elites.csv -g W`

### Removes all user under 50 points

`dendrogram.py examples\results.json examples\elites.csv -t 50`

### Compute data from an specific date interval

`dendrogram.py  examples/results.json examples\elites.csv -i 2021-10-18,2022-10-18`

### Compute data with generic algorithm

`dendrogram.py  examples/results.json examples\elites.csv -l generic

### Compute data with ward method

`dendrogram.py  examples/results.json examples\elites.csv -l ward
