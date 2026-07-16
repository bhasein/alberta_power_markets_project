import cdsapi

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-pressure-levels",
    {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": [
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "relative_humidity",
        ],
        "pressure_level": ["850"],
        "year": "2026",
        "month": "06",
        "day": [
            "01","02","03","04","05","06","07","08","09","10",
            "11","12","13","14","15","16","17","18","19","20",
            "21","22","23","24","25","26","27","28","29","30",
        ],
        "time": [
            "00:00","01:00","02:00","03:00","04:00","05:00",
            "06:00","07:00","08:00","09:00","10:00","11:00",
            "12:00","13:00","14:00","15:00","16:00","17:00",
            "18:00","19:00","20:00","21:00","22:00","23:00",
        ],
        "area": [60.0, -120.5, 48.5, -109.0],
    },
    "data/raw/weather/era5/pressure_levels/2026/06/era5_pressure_850_2026_06.nc",
)