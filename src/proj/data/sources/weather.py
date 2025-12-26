
import pandas as pd

import openmeteo_requests
import requests_cache
from retry_requests import retry
import requests
from requests.exceptions import RequestException
import time
import os

import pandas as pd
import matplotlib.pyplot as plt


DAILY_VAR = ["daylight_duration", "sunshine_duration", "uv_index_max", "uv_index_clear_sky_max", "rain_sum", "showers_sum", "snowfall_sum", "precipitation_hours", "shortwave_radiation_sum", "et0_fao_evapotranspiration", "temperature_2m_mean", "cape_mean", "dew_point_2m_mean", "cloud_cover_mean", "leaf_wetness_probability_mean", "precipitation_probability_mean", "precipitation_sum", "relative_humidity_2m_mean", "pressure_msl_mean", "surface_pressure_mean", "wind_gusts_10m_mean", "wind_speed_10m_mean", "apparent_temperature_mean"]



def get_daily_weather(lats, longs, start, end, variables) -> pd.DataFrame:
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"

    if not isinstance(lats, (list, tuple)):  lats = [lats]
    if not isinstance(longs, (list, tuple)): longs = [longs]
    if isinstance(variables, (list, tuple)):  # <-- key change
        daily_param = ",".join(variables)
    else:
        daily_param = variables  # already a CSV string

    params = {
        "latitude": ",".join(map(str, lats)),
        "longitude": ",".join(map(str, longs)),
        "daily": daily_param,                     # <-- pass CSV
        "start_date": start,
        "end_date": end,
        "timezone": "UTC",
        # optionally: "models": "era5_land"  # good for precip/temps
    }

    responses = openmeteo.weather_api(url, params=params)

    frames = []
    for idx, resp in enumerate(responses):
        daily = resp.Daily()
        data = {
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left",
            ),
            "lat": lats[idx],
            "lon": longs[idx],
        }
        # Map returned arrays in the same order you sent in `variables`
        vars_list = daily_param.split(",")
        for i in range(daily.VariablesLength()):
            name = vars_list[i] if i < len(vars_list) else f"var_{i}"
            data[name] = daily.Variables(i).ValuesAsNumpy()

        frames.append(pd.DataFrame(data))

    return pd.concat(frames, ignore_index=True)


def aggregate_weather(cities, start, end, variables):
    """
    Aggregate weather for all cities in a single Open-Meteo request.
    This is very gentle on the API because it uses one batched call.
    """

    lats  = [c["lat"] for c in cities]
    longs = [c["lon"] for c in cities]

    # One request for all locations
    df = get_daily_weather(lats, longs, start, end, variables)

    # Map (lat, lon) -> city name
    coord_to_name = {(c["lat"], c["lon"]): c["name"] for c in cities}

    df["City_Name"] = [
        coord_to_name.get((lat, lon), None)
        for lat, lon in zip(df["lat"], df["lon"])
    ]

    return df


def fetch():

    return None


def standarize():

    return None


def validate():

    return None