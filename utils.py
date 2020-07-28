from sklearn.preprocessing import MinMaxScaler

def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

def crop_box(image, x, y, w, h):
    if w > h:
        h_diff = int((w - h)/2)
        if y - h_diff > 0:
            y -= h_diff
        img_out = image.crop((x, y, x + w, y + w))
    else:
        w_diff = int((h - w)/2)
        if x - w_diff > 0:
            x -= w_diff
        img_out = image.crop((x, y, x + h, y + h))
    return img_out
