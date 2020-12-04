from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def normalize_data(dr_object, data):

    if dr_object == 'pca':
        """Normalize data between range 0, 1"""
        scaler = MinMaxScaler()
        return scaler.fit_transform(data)

    else:
        """reduce to 300 dimensions"""
        pca = PCA(n_components=300)
        return pca.fit_transform(data)

def crop_box(image, x, y, w, h):
    """Create a centralized squared bounding box based on largest size"""
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
