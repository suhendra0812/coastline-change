# Coastline Change

Coastline change using satellite imagery. In this moment, Sentinel-1 is the only supported satellite imagery which is used to analyze coastline change. Coastline is delineated from binary image which is segmented using image thresholding algorithm (Otsu). Coastline change is performed using transect analysis in time series.

## Requirements

- Python >= 3.7, <= 3.10 (scikit-image is not yet supported in Python 3.11)
- Planetary Computer Subscription Key

## How to get the subscription key:

- Request access to Microsoft Planetary Computer Hub. Please follow this [tutorial](https://planetarycomputer.microsoft.com/docs/overview/environment/).
- Choose **CPU - Python** and click _Start_. After some moments, it will open the jupyter lab environment.
- Open terminal and type the following command.

  ```bash
  $ echo $PC_SDK_SUBSCRIPTION_KEY
  ```

- Copy the key.

## Installation

- Install the `coastline-change` package with the following command.

  ```bash
  $ pip install git+https://github.com/suhendra0812/coastline-change
  ```

- Configure Planetary Computer subscription key with the command below. Paste the key which is copied earlier.
  ```bash
  $ planetarycomputer configure
  Please enter your API subscription key:
  ```

## How to use

### Passing parameters manually in command line interface

- This is the example:
  ```bash
  $ python -m coastline_change \
    --collection sentinel-1-rtc \
    --region-file ./region.json \
    --point-file ./point.json \
    --region-ids 1, 2, 3
    --tide-types mean
    --start-date 2015-01-01
    --end-date 2022-12-31
    --output-dir ./output
    --parallel True
  ```
- Type `python -m coastline_change --help` to see available arguments

### Using config file

- Create `config.json` and passing some parameters that you can see an example in `config.example.json`.
- Run `coastline_change` module with the following command:
  ```bash
  $ python -m coastline_change -f config.json
  ```
