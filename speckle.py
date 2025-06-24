#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import poppy
import astropy.units as u
import logging
logging.basicConfig(level=logging.DEBUG)
from astropy.io import fits
from astropy.modeling import models, fitting
import tifffile
import numpy as np
import matplotlib.colors as mcolors
from astropy.stats import sigma_clip
logging.getLogger().setLevel(logging.INFO)
from scipy.optimize import curve_fit
import photutils
import time
import datetime as dt
from scipy import fftpack
import nbformat
import plotly.graph_objects as go
from PIL import Image
from turbustat.statistics import PowerSpectrum
from scipy.signal import butter, filtfilt
import ipywidgets as widgets
from IPython.display import display, clear_output
from astropy.io.fits import Header
from scipy.signal import find_peaks
import os
import csv
import gc
import re
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus
from photutils.aperture import ApertureStats
from astropy.table import Table
import warnings
from astropy.utils.exceptions import AstropyUserWarning
import winsound
import pandas as pd
from astropy.table import Table


############### PER SPECKLE FIT ###################################
speckle_thresholds=[3000, 4000, 10000]  # l'ultimo era 12000 primada cambiare a seconda dei target usati prima era 3000 non 2000
speckle_threshold=7000 #1100 #7000 #9500 #8000 #9000 #6500 #9000
# Plate scale arcseconds/pixels
plate_scales=[0.0377,  0.0109, 0.012]  #plate cslae di gennaio, febbraio e marzo rispettivamente
plate_scale = 0.0109 # 0.012#4-MARZO #0.0109 #FEBBRAIO# 0.0377 #0.0244*2 #0.0376*2 #0.0268 marzo #0.0244 febb #0.0376 #
# Diffraction limit of the telescope
aperture = 1.82                                                     # Aperture of the telescope (in meters)
wavelength = 633e-9 #700e-9                                                 # Reference wavelenght (in meters) 
diffraction_limit_radians = 2*1.22 * wavelength / aperture          # Diameter of the Airy disk
diffraction_limit_arcseconds = diffraction_limit_radians * 206265
expected_speckle_size_radians = 0.8038 *(1.22 * wavelength / aperture)
expected_speckle_size = expected_speckle_size_radians * 206265  
expected_speckle_size_pixels = expected_speckle_size / plate_scale
# Radius use for the fit of the speckles (in pixels)
radius=int((diffraction_limit_arcseconds/plate_scale)/2)  # raggio della zona del fit attorno alle speckle (Diametro = disco di airy)
check_radius = int((diffraction_limit_arcseconds*0.5)/plate_scale) # raggio della zona di controllo attorno alle speckle (Diametro = disco di airy)
mindist = 5 #distanza dimina in pixel tra 2 speckle (per evitare che la stessa speckle venga contata più volte)
#print("Diffraction limit in arcseconds and pixels: ",diffraction_limit_arcseconds, diffraction_limit_arcseconds/plate_scale)
print("Expected speckle size in arcseconds and pixels: ", expected_speckle_size, expected_speckle_size_pixels)

###########    PER PSD   ##################################################
ordine = 5 #  NON OLTRE 6
imagenumber = 300 #500
stacked = False  # Fa PSD di più immagini sommate (3 al momento)
nstack = 3 # immagini da sommare se si vuole usare il Power specrum di immagini sommate
lowcut_pix = 14#15#7  #max dimension in pixel
highcut_pix = 4#9#2.1  #min dimension in pixel
lowcut= 1/lowcut_pix 
highcut = 1/highcut_pix
crop_size = 2000   # FORMATO IMMAGINE FINALE: N pixel X N pixel (dimiuisce la dimensione dell'immagine per velocizzare i calcoli)
frequency_range = 0.15    # range per trovare il picco delle spckle (centrato nella frequenza delle speckle teorica)
########################################################

def calcProcessTime(starttime, cur_iter, max_iter):

    telapsed = time.time() - starttime
    testimated = (telapsed/cur_iter)*(max_iter)
    finishtime = starttime + testimated
    finishtime = dt.datetime.fromtimestamp(finishtime).strftime("%H:%M:%S")
    lefttime = testimated-telapsed 

    return (int(telapsed), int(lefttime), finishtime)
######################################################################################################################
######################################################################################################################

def fit_speckle_tot(data, filtered_speckles, radius, speckle_threshold, plate_scale):
    fwhm_results = []
    centers = []
    
    for speckle in filtered_speckles:
        y_ref, x_ref = speckle  
        masked_data = data.copy() 
        y, x = np.mgrid[y_ref-radius:y_ref+radius+1, x_ref-radius:x_ref+radius+1]
        masked_data = masked_data[y_ref-radius:y_ref+radius+1, x_ref-radius:x_ref+radius+1]
        masked_data[masked_data < 0] = 0 
        
        ###########################################################################
        #Se attivo, fa il fit in una regione circolare
        distance = np.sqrt((x - x_ref)**2 + (y - y_ref)**2)
        circular_mask = distance <= radius
        masked_data = np.where(circular_mask, masked_data, 0)
        ##############################################################################
        
        # Set border pixels to 0
        masked_data[0, :] = 0
        masked_data[-1, :] = 0
        masked_data[:, 0] = 0
        masked_data[:, -1] = 0

        gaussian_model = models.Gaussian2D(amplitude=masked_data.max(), x_mean=x_ref, y_mean=y_ref, x_stddev=0.5, y_stddev=0.5)
        gaussian_model.amplitude.min = speckle_threshold
        gaussian_model.amplitude.max = masked_data.max()
        gaussian_model.x_mean.min = x_ref - 5   # METTERE A 1 PER LE SPECKLE A 3X
        gaussian_model.x_mean.max = x_ref + 5
        gaussian_model.y_mean.min = y_ref - 5
        gaussian_model.y_mean.max = y_ref + 5

        fitter = fitting.LevMarLSQFitter()
        fitted_model = fitter(gaussian_model, x, y, masked_data)

        fwhm_x = 2.355 * fitted_model.x_stddev.value * plate_scale
        fwhm_y = 2.355 * fitted_model.y_stddev.value * plate_scale
        fwhm_results.append((fwhm_y, fwhm_x))
        centers.append((fitted_model.y_mean.value, fitted_model.x_mean.value))

    return np.array(fwhm_results), np.array(centers)
######################################################################################################################
######################################################################################################################

def gaussiana(bins, media, sigma):
	x = np.zeros(len(bins)-1)
	for i in range(len(x)):
		x[i] = (bins[i]+bins[i+1])/2
	y = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-media)**2/(2*sigma**2))
	return x, y
######################################################################################################################
######################################################################################################################

def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)

######################################################################################################################
######################################################################################################################

def butterworth_2d_bandpass(image, lowcut, highcut, order=5):
    ny, nx = image.shape
    y, x = np.ogrid[:ny, :nx]
    cy, cx = ny // 2, nx // 2
    # Create normalized radius grid (distance from center)
    radius = np.sqrt((x - cx)**2 + (y - cy)**2)
    radius /= np.max(radius)  # Normalize to [0, 1]
    # Normalize lowcut/highcut to [0, 1] as fraction of Nyquist
    low = lowcut * 2
    high = highcut * 2
    # Butterworth bandpass formula
    def butterworth(freq, cutoff, n):
        return 1 / (1 + (freq / cutoff)**(2 * n))
    # Bandpass = Highpass * Lowpass
    lowpass = butterworth(radius, high, order)
    highpass = 1 - butterworth(radius, low, order)
    bandpass_mask = lowpass * highpass
    # Apply filter in frequency domain
    fft_image = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_image)
    filtered_fft = fft_shifted * bandpass_mask
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real
    return filtered_image, bandpass_mask
######################################################################################################################
######################################################################################################################

def crop_center(image, size):
    """
    Crop a square region of given size around the image center.

    """
    ny, nx = image.shape
    cx, cy = nx // 2, ny // 2
    half = size // 2
    x_min = cx - half
    x_max = cx + half
    y_min = cy - half
    y_max = cy + half
    return image[y_min:y_max, x_min:x_max]

######################################################################################################################
######################################################################################################################

def plot_psd_peak(freqs, ps1D, largest_peak_freq, frequency_range, expected_speckle_size, plate_scale):
    zoom_range = (largest_peak_freq - frequency_range / 2, largest_peak_freq + frequency_range / 2)
    zoom_indices = (freqs >= zoom_range[0]) & (freqs <= zoom_range[1])
    zoom_freqs = freqs[zoom_indices]
    zoom_power = ps1D[zoom_indices]
    plt.figure(figsize=(8, 6))
    plt.loglog(zoom_freqs, zoom_power, label='Zoomed Power Spectrum')  # Emphasize the zoomed region
    plt.axvline(x=largest_peak_freq, color='red', linestyle='--', label=f'Peak: {largest_peak_freq:.4f} pix⁻¹')
    plt.axvline(x=1/(expected_speckle_size/plate_scale), color='magenta', linestyle='--', label='Expected Speckle Size')
    plt.title("Zoomed View of Power Spectrum near Largest Peak")
    plt.xlabel("Spatial Frequency [pix⁻¹]")
    plt.ylabel("Power")
    plt.title("Zoomed View of Power Spectrum near Largest Peak")
    plt.grid(True, which="both", linestyle='--')
    plt.legend()
    plt.show()
    
##################################################################################
######################################################################################
def find_peak_power(freqs, ps1D, expected_speckle_size_pixels, frequency_range):
    """
    frequency_range : Range around the expected speckle frequency to search for the peak.
    largest_peak_freq : Frequency with the largest power within the specified range.
    largest_peak_power : Largest power within the specified frequency range.
    """
    expected_speckle_frequency = 1 / expected_speckle_size_pixels
    lower_freq = expected_speckle_frequency - frequency_range / 2
    upper_freq = expected_speckle_frequency + frequency_range / 2
    #print(expected_speckle_size_pixels,expected_speckle_frequency, lower_freq, upper_freq)
    peak_indices = (freqs >= lower_freq) & (freqs <= upper_freq)

    if not any(peak_indices):
        print("No frequencies selected within the specified range.")
        return None, None

    selected_freqs = freqs[peak_indices]
    selected_power = ps1D[peak_indices]
    largest_peak_index = np.argmax(selected_power)
    largest_peak_freq = selected_freqs[largest_peak_index]
    largest_peak_power = selected_power[largest_peak_index]
    
    return largest_peak_freq, largest_peak_power
#################################################################################
##################################################################################
def scale_and_speckle_selector(plate_scales, speckle_thresholds, file):
    if 'arcturus' in file:
        speckle_threshold = speckle_thresholds[1]
    if 'castor' in file:
        speckle_threshold = speckle_thresholds[0]
    if 'aldebaran' in file:
        speckle_threshold = speckle_thresholds[2]   
    if 'febbraio' in file:
        plate_scale = plate_scales[1]
    if 'marzo' in file:
        plate_scale = plate_scales[2]
    print("Plate scale in arcseconds/pixel: ", plate_scale)
    print("Speckle threshold: ", speckle_threshold)
    return plate_scale, speckle_threshold
#######################################################################################################################
#######################################################################################################################
def code_end_alert():
    duration = 5000  # milliseconds
    freqs = [440, 550, 660, 770, 880, 990]  # A few different frequencies
    for freq in freqs:
        winsound.Beep(freq, duration // len(freqs))  # Divide duration to keep total time similar
    freqs = freqs[:-1]
    for freq in reversed(freqs):
        winsound.Beep(freq, duration // (len(freqs)+1))
    print("Code execution completed")
########################################################################################################
#######################################################################################################
warnings.filterwarnings('ignore', category=UserWarning, module='astropy')


folder_path = r"C:\Users\buonc\OneDrive - Alma Mater Studiorum Università di Bologna\Documents\Astrophysics and Cosmology\TESI\dati\marzo\4-03"
file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tif')]
#immagini_list = []

all_dfs = []  # lista per accumulare i DataFrame di ogni file
raw_spectra_all_tot = []

print(f"Found {len(file_list)} files in the folder")
for file in file_list: 
    #file = r"C:\Users\buonc\OneDrive - Alma Mater Studiorum Università di Bologna\Documents\Astrophysics and Cosmology\TESI\dati\marzo\4-03\castor_marzo_40ms_elev75.tif"
    file_name = os.path.basename(file)
    immagini = tifffile.imread(file)[:2000]  #immagini_list.append(immagini)
    print(f"Loaded data from {file_name}")
    print(f'Analysing {file_name}...')
    
    plate_scale, speckle_threshold = scale_and_speckle_selector(plate_scales, speckle_thresholds, file_name)
    diffraction_limit_radians = 2*1.22 * wavelength / aperture          # Diameter of the Airy disk
    diffraction_limit_arcseconds = diffraction_limit_radians * 206265
    expected_speckle_size_radians = 0.8038 *(1.22 * wavelength / aperture)
    expected_speckle_size = expected_speckle_size_radians * 206265  
    expected_speckle_size_pixels = expected_speckle_size / plate_scale
    # Radius use for the fit of the speckles (in pixels)
    radius=int((diffraction_limit_arcseconds/plate_scale)/2)  # raggio della zona del fit attorno alle speckle (Diametro = disco di airy)
    check_radius = int((diffraction_limit_arcseconds*0.5)/plate_scale) # raggio della zona di controllo attorno alle speckle (Diametro = disco di airy)
    #mindist = 5 #distanza dimina in pixel tra 2 speckle (per evitare che la stessa speckle venga contata più volte)
    
    # ############# SPECKLE FIT ##########################################################
    
    # start = time.time()
    # fwhm_single_image = []
    # fwhm_mean_all = []
    # fwhm_median_all = []
    # fwhm_std_all = []
    # centroid_all = []   # first coordinate in the array is the Y
    # rms_all = []        # first coordinate in the array is the Y
    # fwhm_mean_filtered_all = []
    # fwhm_median_filtered_all = []
    # fwhm_std_filtered_all = []
    # rms_filtered_all = []
    # centroid_filtered_all = []
    # ellipticity_mean_all=[]

    # for imagenumber in range(len(immagini)):
    #     data_raw = immagini[imagenumber]
    #     data_clean = data_raw
    #     poisson_error = np.sqrt(data_raw)
    #     background_level = np.median(data_raw)
    #     background_error = np.sqrt(background_level)
    #     noise = np.sqrt(np.mean(poisson_error**2) + background_error**2)
    #     background_estimate = data_raw - noise
    #     background_estimate[background_estimate < 0] = 0
    #     data = data_raw - background_level
    #     data[data < 0] = 0

    #     ###########################     SOLO PER 10KHZ ################################################
    #     #center_y, center_x = data.shape[0] // 2, data.shape[1] // 2
    #     #search_radius = 80  ###per speckle a 3x
    #     #y, x = np.ogrid[:data.shape[0], :data.shape[1]]
    #     #mask = (x - center_x)**2 + (y - center_y)**2 <= search_radius**2
    #     #speckle_coords = np.column_stack(np.where((data > speckle_threshold) & mask))
    #     ################################################################################################
    #     # QUANDO NON ATTIVO SPECKLE A 10KHZ
    #     speckle_coords = np.column_stack(np.where(data > speckle_threshold))
    #     real_speckles = []

    #     for coord in speckle_coords:
    #         y, x = coord
    #         max_count = data[y, x]
    #         speckle = coord
    #         # Check the surrounding pixels within the radius
    #         for dy in range(-check_radius, check_radius + 1):
    #             for dx in range(-check_radius, check_radius + 1):
    #                 ny, nx = y + dy, x + dx
    #                 if 0 <= ny < data.shape[0] and 0 <= nx < data.shape[1]:
    #                     if data[ny, nx] > max_count:
    #                         max_count = data[ny, nx]
    #                         speckle = [ny, nx]           
    #         if not any(np.array_equal(speckle, x) for x in real_speckles):
    #             real_speckles.append(speckle)
                
    #     real_speckles = np.array(real_speckles)
    #     filtered_speckles = []
                
    #     for speckle in real_speckles:
    #         y, x = speckle
    #         max_count = data[y, x]
    #         keep_speckle = True
    #         # Check for speckles that are closer than N pixels
    #         for other_speckle in filtered_speckles:
    #             distance = np.sqrt((other_speckle[0] - y)**2 + (other_speckle[1] - x)**2)
    #             if distance < mindist:   # change here to change the radius (in pixels)
    #                 if data[other_speckle[0], other_speckle[1]] > max_count:
    #                     keep_speckle = False
    #                 else:
    #                     filtered_speckles = [fs for fs in filtered_speckles if not np.array_equal(fs, other_speckle)]
    #         if keep_speckle:
    #             filtered_speckles.append([y, x])

    #     filtered_speckles = np.array(filtered_speckles)
    #     h, w = data.shape
    #     # solo speckle tali che  [y-radius:y+radius+1, x-radius:x+radius+1] stia  dentro l’immagine
    #     ys = filtered_speckles[:, 0]
    #     xs = filtered_speckles[:, 1]
    #     mask = (
    #         (ys - radius >= 0) &
    #         (ys + radius + 1 <= h) &
    #         (xs - radius >= 0) &
    #         (xs + radius + 1 <= w)
    #     )
    #     filtered_speckles = filtered_speckles[mask]
    # ###################################à SOLO PER 10KHZ ##############################################  
    #     # Calculate the barycenter of the filtered speckles
    #     #barycenter_x = filtered_speckles[:, 1].mean()
    #     #barycenter_y = filtered_speckles[:, 0].mean()
    # # Define a maximum distance from the barycenter to consider a speckle as valid
    #     #max_distance = 50  
    #     #distances = np.sqrt((filtered_speckles[:, 1] - barycenter_x)**2 + (filtered_speckles[:, 0] - barycenter_y)**2)
    #     #filtered_speckles = filtered_speckles[distances <= max_distance] 
    # #############################################################################

       
    #     fwhm_single_image, centers = fit_speckle_tot(data, filtered_speckles, radius, speckle_threshold, plate_scale)
    #     #print(f"Number of speckles found in image {imagenumber}: {len(filtered_speckles)}")
        
    #     ellipticity_single_image = 1 - np.minimum(fwhm_single_image[:, 0], fwhm_single_image[:, 1]) / np.maximum(fwhm_single_image[:, 0], fwhm_single_image[:, 1])

    #     #tutto con indice 1 perché si sta facendo statistica solo sulla X
    #     fwhm_x_clipped = sigma_clip(fwhm_single_image[:, 1], sigma=3, maxiters=1)  # o 1 a 3 sigma o 3 a 3.5 sigma
    #     fwhm_x_tot_clean = fwhm_single_image[~fwhm_x_clipped.mask]
    #     mean_clipped_fwhm = np.mean(fwhm_x_tot_clean[:,1])
    #     median_clipped_fwhm = np.ma.median(fwhm_x_tot_clean[:,1])
    #     std_clipped_fwhm = np.std(fwhm_x_tot_clean[:,1]) 
    #     centroid = (np.mean(centers[:,0]), np.mean(centers[:,1])) 
    #     rms = (np.sqrt(np.mean((centers[:,0] - centroid[0])**2)), np.sqrt(np.mean((centers[:,1] - centroid[1])**2))) 
    #     # mean_clipped_fwhm = np.mean(fwhm_x_tot_clean)
    #     # median_clipped_fwhm = np.ma.median(fwhm_x_tot_clean)
    #     # std_clipped_fwhm = np.std(fwhm_x_tot_clean)

    #     # Store results for all speckles
    #     fwhm_mean_all.append(mean_clipped_fwhm)
    #     fwhm_median_all.append(median_clipped_fwhm)
    #     fwhm_std_all.append(std_clipped_fwhm)
    #     rms_all.append(rms)
    #     centroid_all.append(centroid)
    #     ellipticity_mean_all.append(np.mean(ellipticity_single_image))
        
    #     if imagenumber % 500 == 0:
    #         prstime = calcProcessTime(start, imagenumber +1 ,len(immagini))
    #         print("time elapsed: %s s, time left: %s s, estimated finish time: %s"%prstime)
            
    # print('Speckles fit done')

######################## SPECKLE PSD ######################

    print(f'Power Spectra of{file_name}...')
    
    header = Header()
    header['CDELT1'] = 1.0
    header['CDELT2'] = 1.0
    peak_frequencies = []
    peak_powers = []
    raw_spectra_all = []
    for img in range(len(immagini)):

        print("processing image ", img ,'/', len(immagini))
        largest_peak_freq = 0
        largest_peak_power = 0
        
        #if stacked and i + nstack <= len(immagini):
        #    data = np.sum(immagini[i:i + 3], axis=0)
        #else:
        data = immagini[img]
        image_cropped = crop_center(data, crop_size)
        data = image_cropped.copy()
        background_level = np.median(data)
        data = data - background_level
        data[data < 0] = 0
        image = data.copy()

        # Non fitered power spectrum
        pspec_raw = PowerSpectrum(image, header=header, distance=1 * u.pc)
        pspec_raw.run(verbose=False, xunit=u.pix**-1)
        freqs_raw = pspec_raw.freqs.value
        psd1D_raw = pspec_raw.ps1D
        raw_spectra_all.append({
            'file': file_name,
            'image_index': img,
            'freqs': freqs_raw,
            'powers': psd1D_raw
        })
        
        # Apply 2D Butterworth bandpass
        filtered_image, mask = butterworth_2d_bandpass(image, lowcut, highcut, ordine)
        # Normalize filtered image
        filtered_image = np.nan_to_num(filtered_image, nan=0.0, posinf=0.0, neginf=0.0)
        filtered_image[filtered_image < 0] = 0
        filtered_image /= (np.max(filtered_image) + 1e-8)

        # Compute power spectrum
        pspec_filtered = PowerSpectrum(filtered_image, header=header, distance=1 * u.pc)
        pspec_filtered.run(verbose=False, xunit=u.pix**-1)
        
        freqs = pspec_filtered.freqs.value
        ps1D = pspec_filtered.ps1D
        largest_peak_freq, largest_peak_power = find_peak_power(freqs, ps1D, expected_speckle_size_pixels, frequency_range)
        
        peak_frequencies.append(largest_peak_freq)
        peak_powers.append(largest_peak_power)

        print('peak frequencies:', peak_frequencies)
        print('peak powers:', peak_powers)

        if largest_peak_freq == 0 and largest_peak_power == 0:
             print("Could not find any peaks within the specified frequency range.")
        print(f'Power Spectrum of {file_name} done')
        
############################# SALVATAGGIO RISULTATI ########################################
#######################################################################################     

    
    #centroid_x = [c[1] for c in centroid_all]
    #centroid_y = [c[0] for c in centroid_all]
    #rms_x = [r[1] for r in rms_all]
    #rms_y = [r[0] for r in rms_all]
    
    data = {
    #'FILE':             file_name,
    #'FWHM_MEAN':        fwhm_mean_all,
    #'FWHM_MEDIAN':      fwhm_median_all,
    #'FWHM_STD':         fwhm_std_all,
    #'CENTROID_X':       centroid_x,
    #'CENTROID_Y':       centroid_y,
    #'RMS_X':            rms_x,
    #'RMS_Y':            rms_y,
    #'MEAN_ELLIPTICITY': ellipticity_mean_all
    'PEAK_FREQUENCY':    peak_frequencies,
    'PEAK_POWER':        peak_powers,
    }
    df = pd.DataFrame(data)
    
    all_dfs.append(df)
    raw_spectra_all_tot.extend(raw_spectra_all)
    
    ######################## PER FARE TUTTI I RISULTATI IN UN UNICO FILE
    #all_dfs.append(df)

    del immagini
    gc.collect()

############### PER TUTTI I RISULTATI IN UN FILE SOLO
# Dopo aver processato tutti i file, concatena tutti i DataFrame
df_all = pd.concat(all_dfs, ignore_index=True)
# Crea la cartella di output (se non esiste)
os.makedirs('outputs', exist_ok=True)
# Salva CSV unico con tutti i risultati
output_csv_all = 'outputs/all_results.csv'
df_all.to_csv(output_csv_all, index=False)
print(f"All results saved to CSV file: {output_csv_all}")
# Salva FITS unico con tutti i risultati
output_fits_all = 'outputs/all_results.fits'
Table.from_pandas(df_all).write(output_fits_all, format='fits', overwrite=True)
print(f"All results saved to FITS table: {output_fits_all}")



rows = []
for entry in raw_spectra_all_tot:
    fn = entry['file']
    idx = entry['image_index']
    freqs = entry['freqs']
    powers = entry['powers']

    df_tmp = pd.DataFrame({
        'file': fn,
        'image_index': idx,
        'frequency': freqs,
        'power': powers
    })
    rows.append(df_tmp)

df_long = pd.concat(rows, ignore_index=True)
print(f"Totale righe PSD raw nel df: {len(df_long)}")

csv_path = os.path.join('outputs', 'all_raw_psd.csv')
df_long.to_csv(csv_path, index=False)
print(f"Saved raw PSD long CSV to {csv_path}")

# parquet_path = os.path.join('outputs', 'all_raw_psd.parquet')
# df_long.to_parquet(parquet_path, index=False)
# print(f"Saved raw PSD parquet to {parquet_path}")

# hdf_path = os.path.join('outputs', 'all_raw_psd.h5')
# df_long.to_hdf(hdf_path, key='raw_psd', mode='w')
# print(f"Saved raw PSD HDF5 to {hdf_path}")