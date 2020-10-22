import numpy as np

# given a probmap, an original windowsize, an original stride length, and an array specifying the size of window
# associated with each non-zero value in the probability map, returns the coordinates of the maximum window and its
# size
def nms(probmap, windowsize, stride, windows):
    height, width = probmap.shape
    face_i = []
    face_j = []
    window_i_j = []

    windowsize = int(windowsize/2)
    for i in range(0, height-windowsize, stride):
        for j in range(0, width - windowsize, stride):
            crop = probmap[i:i+windowsize, j:j+windowsize]
            max = np.max(crop)
            crop[crop < max] = 0
            probmap[i:i + windowsize, j:j + windowsize] = crop

    for i in range(0, height):
        for j in range(0, width):
            if probmap[i][j] > 0:
                face_i.append(i)
                face_j.append(j)
                window_i_j.append(int(windows[i][j]))
    remove = []
    for i in range(len(face_i)):
        for j in range(len(face_i)):
            if j != i:
                distance = (face_i[i] - face_i[j]) ** 2 + (face_j[i] - face_j[j]) ** 2
                if distance < (window_i_j[i]**2)*(1/2):
                    if probmap[face_i[i], face_j[i]] > probmap[face_i[j], face_j[j]]:
                        remove.append(j)
                    else:
                        remove.append(i)

    new_i = []
    new_j = []
    new_wind = []

    for i in range(len(face_i)):
        if i not in remove:
            new_i.append(face_i[i])
            new_j.append(face_j[i])
            new_wind.append(window_i_j[i])
    face_i = new_i
    face_j = new_j
    window_i_j = new_wind
    return face_i, face_j, window_i_j
