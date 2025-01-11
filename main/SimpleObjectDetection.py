import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.morphology import remove_small_objects, dilation, disk
from matplotlib.widgets import Slider
from tkinter import Tk, filedialog

def process_image(image, canny_low, canny_high, threshold_val, min_area, max_area):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Buat gambar hitam putih
    edges = cv2.Canny(gray_image, canny_low, canny_high)  # Aplikasikan Canny edge detection
    binary_image = gray_image < threshold_val  # Buat binary mask dari thresholding
    dilated_edges = dilation(edges > 0, disk(1))  # Dilated edges to create edge regions
    combined_mask = binary_image & dilated_edges  # Gabungkan informasi tepi dengan threshold
    cleaned_mask = remove_small_objects(combined_mask, min_size=min_area)

    # Cari dan filter region
    labels = measure.label(cleaned_mask)
    props = measure.regionprops(labels)

    # Buat gambar keluaran
    detected_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    final_mask = np.zeros_like(binary_image)
    object_detected = False

    for prop in props:
        if min_area <= prop.area <= max_area:
            minr, minc, maxr, maxc = prop.bbox
            cv2.rectangle(detected_image, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
            final_mask[minr:maxr, minc:maxc] = prop.image
            object_detected = True

    return gray_image, edges, final_mask, detected_image, object_detected

def interactive_detection():
    # Pilih file gambar
    Tk().withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(
        title="Pilih gambar",
        filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")]
    )

    if not file_paths:
        print("Tidak ada gambar yang dipilih.")
        return

    for image_path in file_paths:
        image = cv2.imread(image_path)

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 3, hspace=0.4, wspace=0.4)
    ax = [fig.add_subplot(gs[0, 0]),
          fig.add_subplot(gs[0, 1]),
          fig.add_subplot(gs[0, 2])]

    fig.suptitle('Bayu Ganteng', fontsize=16)

    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.25, top=0.9)

    # Parameter awal
    init_canny_low = 100
    init_canny_high = 200
    init_threshold_val = 140
    init_min_area = 50
    init_max_area = 200

    # Proses gambar yang dipilih dan simpan di variabel
    gray, edges, binary, detected, object_detected = process_image(
        image, init_canny_low, init_canny_high, init_threshold_val, init_min_area, init_max_area
    )

    # Tampilkan gambar
    img_edges = ax[0].imshow(edges, cmap='gray')
    ax[0].set_title('Canny Edge Detection')

    img_binary = ax[1].imshow(binary)
    ax[1].set_title('Combined Edge-Threshold Mask')

    img_detected = ax[2].imshow(detected)
    ax[2].set_title('Detected Object: ' + ('Yes' if object_detected else 'No'))

    # Buat slider untuk atur parameter
    slider_y = [0.15, 0.12, 0.09, 0.06, 0.03]
    slider_width = 0.6
    slider_height = 0.02

    ax_canny_low = plt.axes([0.15, slider_y[0], slider_width, slider_height])
    ax_canny_high = plt.axes([0.15, slider_y[1], slider_width, slider_height])
    ax_threshold = plt.axes([0.15, slider_y[2], slider_width, slider_height])
    ax_min_area = plt.axes([0.15, slider_y[3], slider_width, slider_height])
    ax_max_area = plt.axes([0.15, slider_y[4], slider_width, slider_height])

    s_canny_low = Slider(ax_canny_low, 'Canny Low', 0, 255, valinit=init_canny_low)
    s_canny_high = Slider(ax_canny_high, 'Canny High', 0, 255, valinit=init_canny_high)
    s_threshold = Slider(ax_threshold, 'Threshold', 0, 255, valinit=init_threshold_val)
    s_min_area = Slider(ax_min_area, 'Min Area', 0, 200, valinit=init_min_area)
    s_max_area = Slider(ax_max_area, 'Max Area', 200, 1000, valinit=init_max_area)

    def update(val=None):
        gray, edges, binary, detected, object_detected = process_image(
            image,
            s_canny_low.val,
            s_canny_high.val,
            s_threshold.val,
            s_min_area.val,
            s_max_area.val
        )
        img_edges.set_data(edges)
        img_binary.set_data(binary)
        img_detected.set_data(detected)
        ax[2].set_title('Detected Object: ' + ('Yes' if object_detected else 'No'))

        fig.canvas.draw_idle()

    s_canny_low.on_changed(update)
    s_canny_high.on_changed(update)
    s_threshold.on_changed(update)
    s_min_area.on_changed(update)
    s_max_area.on_changed(update)

    plt.show()

interactive_detection()
