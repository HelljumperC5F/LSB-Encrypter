"""
# My first app
Here's our first attempt at using data to create a table:

To run the app use 'streamlit run ./main.py'
"""

import streamlit as st
import numpy as np
import skimage
import io


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def psnr(imageA, imageB):
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse_value))


def clicked_hide():
    try:
        cover_bytes = uploader_cover.read()
    except AttributeError:
        st.error('You did not upload a cover image.')
        return

    try:
        secret_bytes = uploader_secret.read()
    except AttributeError:
        st.error('You did not upload a secret image.')
        return

    cover_image = skimage.io.imread(io.BytesIO(cover_bytes))
    shape = [cover_image.shape[0], cover_image.shape[1]]

    secret_image = skimage.io.imread(io.BytesIO(secret_bytes))
    secret_image = skimage.color.rgb2gray(secret_image)
    secret_image = skimage.transform.resize(secret_image, shape)
    secret_image = secret_image * 255
    secret_image = np.floor(secret_image)
    secret_image = np.array(secret_image, int)
    # secret_image = secret_image >> 8 - bit_slider
    #
    # secret_r = (secret_image >> 4) & 0b00000011
    # secret_g = (secret_image >> 2) & 0b00000011
    # secret_b = (secret_image & 0b00000011)

    match bit_slider:
        case 1:
            secret_r = None
            secret_g = None
            secret_b = (secret_image >> 7) & 0b00000001

            cover_image[:, :, 2] &= 0b11111110
        case 2:
            secret_r = (secret_image >> 7) & 0b00000001
            secret_g = None
            secret_b = (secret_image >> 6) & 0b00000001

            cover_image[:, :, 0] &= 0b11111110
            cover_image[:, :, 2] &= 0b11111110
        case 3:
            secret_r = (secret_image >> 7) & 0b00000001
            secret_g = (secret_image >> 6) & 0b00000001
            secret_b = (secret_image >> 5) & 0b00000001

            cover_image[:, :, 0] &= 0b11111110
            cover_image[:, :, 1] &= 0b11111110
            cover_image[:, :, 2] &= 0b11111110
        case 4:
            secret_r = (secret_image >> 7) & 0b00000001
            secret_g = (secret_image >> 6) & 0b00000001
            secret_b = (secret_image >> 4) & 0b00000011

            cover_image[:, :, 0] &= 0b11111110
            cover_image[:, :, 1] &= 0b11111110
            cover_image[:, :, 2] &= 0b11111100
        case 5:
            secret_r = (secret_image >> 6) & 0b00000011
            secret_g = (secret_image >> 5) & 0b00000001
            secret_b = (secret_image >> 3) & 0b00000011

            cover_image[:, :, 0] &= 0b11111100
            cover_image[:, :, 1] &= 0b11111110
            cover_image[:, :, 2] &= 0b11111100
        case 6:
            secret_r = (secret_image >> 6) & 0b00000011
            secret_g = (secret_image >> 4) & 0b00000011
            secret_b = (secret_image >> 2) & 0b00000011

            cover_image[:, :, 0] &= 0b11111100
            cover_image[:, :, 1] &= 0b11111100
            cover_image[:, :, 2] &= 0b11111100
        case 7:
            secret_r = (secret_image >> 6) & 0b00000011
            secret_g = (secret_image >> 4) & 0b00000011
            secret_b = (secret_image >> 1) & 0b00000111

            cover_image[:, :, 0] &= 0b11111100
            cover_image[:, :, 1] &= 0b11111100
            cover_image[:, :, 2] &= 0b11111000
        case 8:
            secret_r = (secret_image >> 5) & 0b00000111
            secret_g = (secret_image >> 3) & 0b00000011
            secret_b = secret_image & 0b00000111

            cover_image[:, :, 0] &= 0b11111000
            cover_image[:, :, 1] &= 0b11111100
            cover_image[:, :, 2] &= 0b11111000

    # st.write(format(secret_image[0][0], '08b'))
    #
    # st.write(format(secret_r[0][0], '08b'))
    # st.write(format(secret_g[0][0], '08b'))
    # st.write(format(secret_b[0][0], '08b'))

    # cover_image = cover_image & 0b11111100
    if secret_r is not None:
        cover_image[:, :, 0] = cover_image[:, :, 0] | secret_r
    if secret_g is not None:
        cover_image[:, :, 1] = cover_image[:, :, 1] | secret_g
    if secret_b is not None:
        cover_image[:, :, 2] = cover_image[:, :, 2] | secret_b

    decoded_image = np.empty(shape, int)
    decoded_image = 0b00000000

    match bit_slider:
        case 1:
            decoded_r = None
            decoded_g = None
            decoded_b = (cover_image[:, :, 2] & 0b00000001) << 7
        case 2:
            decoded_r = (cover_image[:, :, 0] & 0b00000001) << 7
            decoded_g = None
            decoded_b = (cover_image[:, :, 2] & 0b00000001) << 6
        case 3:
            decoded_r = (cover_image[:, :, 0] & 0b00000001) << 7
            decoded_g = (cover_image[:, :, 1] & 0b00000001) << 6
            decoded_b = (cover_image[:, :, 2] & 0b00000001) << 5
        case 4:
            decoded_r = (cover_image[:, :, 0] & 0b00000001) << 7
            decoded_g = (cover_image[:, :, 1] & 0b00000001) << 6
            decoded_b = (cover_image[:, :, 2] & 0b00000011) << 4
        case 5:
            decoded_r = (cover_image[:, :, 0] & 0b00000011) << 6
            decoded_g = (cover_image[:, :, 1] & 0b00000001) << 5
            decoded_b = (cover_image[:, :, 2] & 0b00000011) << 3
        case 6:
            decoded_r = (cover_image[:, :, 0] & 0b00000011) << 6
            decoded_g = (cover_image[:, :, 1] & 0b00000011) << 4
            decoded_b = (cover_image[:, :, 2] & 0b00000011) << 2
        case 7:
            decoded_r = (cover_image[:, :, 0] & 0b00000011) << 6
            decoded_g = (cover_image[:, :, 1] & 0b00000011) << 4
            decoded_b = (cover_image[:, :, 2] & 0b00000111) << 1
        case 8:
            decoded_r = (cover_image[:, :, 0] & 0b00000111) << 5
            decoded_g = (cover_image[:, :, 1] & 0b00000011) << 3
            decoded_b = cover_image[:, :, 2] & 0b00000111

    decoded_gray = 0b00000000
    if decoded_r is not None:
        decoded_gray |= decoded_r
    if decoded_g is not None:
        decoded_gray |= decoded_g
    if decoded_b is not None:
        decoded_gray |= decoded_b

    decoded_image = decoded_gray

    st.success('Image Encrypted!')
    st.subheader('Metricas de error')
    st.text(f'MSE: {mse(secret_image, decoded_image)}')
    st.text(f'PSNR: {psnr(secret_image, decoded_image)}')
    st.header('Encrypted Image')
    st.image(cover_image)
    st.header('Decrypted Secret Image')
    st.image(decoded_image)


st.title('LSB Steganography Encrypter')

bit_slider = st.slider(
        label='Number of bits to use on the cover image:',
        min_value=1,
        max_value=8,
        value=6
        )

uploader_cover = st.file_uploader(
        label='Cover Image (The image you want to show)',
        type=['jpg', 'jpeg']
        )

uploader_secret = st.file_uploader(
        label='Secret Image (The image you want to hide)',
        type=['jpg', 'jpeg']
        )

st.button(
        label='Hide',
        type='primary',
        use_container_width=True,
        on_click=clicked_hide)
