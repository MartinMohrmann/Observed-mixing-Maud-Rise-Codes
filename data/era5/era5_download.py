import cdsapi
import os

c = cdsapi.Client()

#for year in ['2018', '2019', '2020', '2017', '2016']:
for year in ['2021']:
    for month in ['01', '02', '03', '04', '05', '06',
                  '07', '08', '09', '10', '11', '12',]:
        #if os.path.isfile('download_%s_%s.nc'%(year, month)):
        #    print('%s-%s already existed'%(year, month))
        #else:
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': [
                        '10m_u_component_of_wind', '10m_v_component_of_wind', '10m_wind_gust_since_previous_post_processing',
                        '2m_temperature', 'forecast_albedo', 'mean_evaporation_rate',
                        'mean_sea_level_pressure', 'mean_surface_direct_short_wave_radiation_flux', 'mean_surface_latent_heat_flux',
                        'mean_surface_sensible_heat_flux', 'sea_ice_cover', 'sea_surface_temperature',
                        'surface_pressure', 'total_precipitation',
                    ],
                    'year': year,
                    'month': month,
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area': [
                        -60, -10, -70,
                        10,
                    ],
                    'format': 'netcdf',
                },
                'download_%s_%s.nc'%(year, month))
