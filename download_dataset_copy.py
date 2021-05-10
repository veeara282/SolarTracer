import hashlib
import hmac
import base64
import requests
import urllib.parse as urlparse
import time
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
# these are commented out but are needed to run (need to use different api key)
api_key = ''
secret = ''

# lat, lng = 51.507574, -0.127835
# lat, lng = 51.5252601, 0.1276249 # first const site
# lat, lng = 51.5222601, 0.1196249 # second const site
lat, lng = 51.6587601, -0.0203751 # third const site
dt_lat, dt_lng = 0.0015, 0.002
def sign_url(input_url=None, secret=None):
    """ Sign a request URL with a URL signing secret.
      Usage:
      from urlsigner import sign_url
      signed_url = sign_url(input_url=my_url, secret=SECRET)
      Args:
      input_url - The URL to sign
      secret    - Your URL signing secret
      Returns:
      The signed request URL
  """

    if not input_url or not secret:
        raise Exception("Both input_url and secret are required")

    url = urlparse.urlparse(input_url)

    # We only need to sign the path+query part of the string
    url_to_sign = url.path + "?" + url.query

    # Decode the private key into its binary format
    # We need to decode the URL-encoded private key
    decoded_key = base64.urlsafe_b64decode(secret)

    # Create a signature using the private key and the URL-encoded
    # string using HMAC SHA1. This signature will be binary.
    signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)

    # Encode the binary signature into base64 for use within a URL
    encoded_signature = base64.urlsafe_b64encode(signature.digest())

    original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query

    # Return signed URL
    return original_url + "&signature=" + encoded_signature.decode()

def convert_params(parameter_dict):
    return_str= ''
    for key, value in parameter_dict.items():
        return_str += f'{key}={value}&'
    return return_str[:-1]

def develop_url(lat, lng):
    url = 'https://maps.googleapis.com/maps/api/staticmap?'
    # DeepSolar paper: 320x320 images at zoom level 21
    params = {
            'center': f'{lat},{lng}',
            'size': '320x320',
            'maptype': 'satellite',
            'zoom':'21',
            'key': api_key
    }
    url = f'{url}{convert_params(params)}'
    return sign_url(url, secret)

def save_img(lat, lng, fn):
    url = develop_url(lat, lng)
    r = requests.get(url)
    f = open(fn, 'wb')
    f.write(r.content)
    f.close()

def start_scraping():
    img_names = set([tuple(x.replace(".png", "").split("_")) for x in os.listdir('outputs/images/')])
    # lat_lon = np.load('outputs/lat_lon_query.npy')
    query = pd.read_csv('outputs/final_query.csv', index_col = 0)
    lat_lon = query[['latitude', 'longitude']].values

    rng = np.random.default_rng()

    start_time = time.time()
    # Keep track of the number of requests
    num_requests = 0
    tqdm_obj = tqdm(range(lat_lon.shape[0]), desc = 'qps: 0')
    for idx in tqdm_obj:
        # Randomly sample from a 0.001 degree square neighborhood around the base poin
        base_lat, base_lon = lat_lon[idx]
        max_dist = 0.001
        n_samples = 5

        # Sample from a Gaussian distribution bounded by [center - max_dist, center + max_dist]
        def random_coords(center):
            samples = rng.normal(center, max_dist / 2, n_samples)
            return np.clip(samples, center - max_dist, center + max_dist, out=samples)

        lats = random_coords(base_lat)
        lons = random_coords(base_lon)

        for i in range(n_samples):
            lat, lon = lats[i], lons[i]
            key = (f"{lat:.6f}", f"{lon:.6f}")
            if key in img_names:
                continue
            save_img(lat, lon, f'outputs/images/{lat:.6f}_{lon:.6f}.png')
            num_requests += 1
            # ensure that the number of requests < 500/s
            if num_requests % 500 == 0:
                time_passed = time.time() - start_time
                if time_passed < (num_requests // 500):
                    time.sleep((num_requests // 500) - time_passed)
                tqdm_obj.set_description(f'qps: {num_requests/time_passed:.4f}')

# query = pd.read_csv('outputs/final_query.csv', index_col = 0)
# embed()
# 1/0
# save_img(lat, lng, 'outputs/const3.png')
start_scraping()
