import numpy as np
from sklearn.cluster import k_means


def division(gb_list, sample_threshold):
    gb_list_new = []
    for gb in gb_list:
        if gb.shape[0] >= sample_threshold:
            ball_1, ball_2 = spilt_ball(gb)  # Split the GB using 2-means
            SD_original = get_SD(gb)  # Calculate the SD of the original GB

            if len(ball_1) == 0:  # The samples were all counted in ball_2
                gb_list_new.append(ball_2)
                continue
            elif len(ball_2) == 0:  # The samples were all counted in ball_1
                gb_list_new.append(ball_1)
                continue

            SD_k_1 = get_SD(ball_1)  # Calculate the SD of sub-GB 1
            SD_k_2 = get_SD(ball_2)  # Calculate the SD of sub-GB 2

            SD_child = SD_k_1 + SD_k_2  # Calculate the SD_child of sub-GBs

            # Splitting criterion
            if SD_child < SD_original:
                gb_list_new.extend([ball_1, ball_2])
            else:
                gb_list_new.append(gb)
        else:
            gb_list_new.append(gb)
    return gb_list_new

# Calculate the Sum Distance SD of GBs.
def get_SD(gb):
    # input:
    # gb is a array including all samples in the gb.
    data = gb[:, :-1]  # Delete serial number column
    center = data.mean(0)
    SD = np.sum(((data - center) ** 2).sum(axis=1) ** 0.5)
    return SD

def spilt_ball(gb):
    data = gb[:, :-1]
    cluster = k_means(X=data, init='k-means++', n_clusters=2)[1]
    ball1 = gb[cluster == 0, :]
    ball2 = gb[cluster == 1, :]
    return [ball1, ball2]


def getGranularBall(data, delta):
    sample_threshold = int(delta * len(data))
    index = np.arange(0, data.shape[0], 1)  # A zero column vector with a size of 1 minus the sample size
    data_index = np.insert(data, data.shape[1], values=index, axis=1)  # add the index of samples
    data_index[:,-1] = data_index[:,-1].astype(int)
    gb_list_temp = [data_index]  # the whole dataset as a GB

    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp = division(gb_list_temp,sample_threshold)  
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:  # no more splitting
            break

    gb_list_final = gb_list_temp

    return gb_list_final
