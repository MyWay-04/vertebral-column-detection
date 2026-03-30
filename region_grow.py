import queue
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def region_reindex(region_img):
    height, width = region_img.shape
    # reindex region number from bottom to top
    reg_idx = [0]
    for row in range(height-1, -1, -1):
        for col in range(width):
            if region_img[row, col] > 0:
                if region_img[row, col] not in reg_idx:
                    reg_idx.append(region_img[row, col])
    reg_map = {v:k for k, v in enumerate(reg_idx)}
    for row in range(height):
        for col in range(width):
            region_img[row, col] = reg_map[region_img[row, col]]
    return region_img

def avg_column_bound(comp):
    height, width = comp.shape
    section_h = (height//15) + 1
    top = -section_h
    shift = int(0.2 * width)
    bottom = 0
    marker = np.max(comp) + 1
    coord = [[] for _ in range(height)]
    while True:
        top += section_h
        bottom += section_h
        avg = 0
        count = 0
        if top >= height:
            break
        for row in range(top, min(bottom, height)):
            for col in range(width):
                if comp[row, col] == 0:
                    continue
                avg += col
                count += 1
        if count == 0:
            avg = width//2
        else:
            avg /= count
            avg = int(avg)
        for row in range(top, min(bottom, height)):
            coord[row] = list(range(avg-shift, avg+shift))

    return coord

def region_grow_cont(img, seed, threshold, vis = False):
    height, width = img.shape
    region = seed.copy()
    visited = np.zeros_like(seed)
    region_idx = np.unique(seed).tolist()
    if 0 in region_idx:
        region_idx.remove(0)
    region_stat = {}
    for r in region_idx:
        region_stat[r] = {'max': np.max(img[seed == r]), 'min': np.min(img[seed == r])}

    q = queue.Queue()
    counter = 0

    for row in range(height):
        for col in range(width):
            if region[row, col] > 0:
                visited[row, col] = 1
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        r = min(max(row + i, 0), height - 1)
                        c = min(max(col + j, 0), width - 1)
                        if region[r, c] == 0 and visited[r, c] == 0:
                            q.put((r, c))
                        visited[r, c] = 1
    while not q.empty():
        row, col = q.get()
        counter += 1
        min_diff = 255
        curr_reg = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                r = min(max(row + i, 0), height - 1)
                c = min(max(col + j, 0), width - 1)
                if region[r, c] == 0:
                    if visited[r, c] == 0:
                        q.put((r, c))
                        visited[r, c] = 1
                else:
                    diff = min(abs(img[row, col] - region_stat[region[r, c]]['max']), abs(img[row, col] - region_stat[region[r, c]]['min']))
                    if diff < min_diff:
                        min_diff = diff
                        curr_reg = region[r, c]
        if min_diff < threshold:
            region[row, col] = curr_reg

    return region

def region_grow_iter(img, seed, threshold, vis = False):
    height, width = img.shape
    region = seed.copy()
    region_idx = np.unique(seed).tolist()
    region_idx.remove(0)
    region_stat = {}
    for r in region_idx:
        region_stat[r] = {'max': np.max(img[seed == r]), 'min': np.min(img[seed == r])}

    coords = avg_column_bound(seed)

    for iter in range(1000):
        change = 0
        for row in range(height):
            for col in coords[row]:
                if region[row, col] == 0:
                    nb_list = []
                    nb_diff = []
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            if i == 0 and j == 0:
                                continue
                            r = min(max(row + i, 0), height - 1)
                            c = min(max(col + j, 0), width - 1)
                            if region[r, c] in region_idx:
                                curr_reg = region[r, c]
                                curr_min = region_stat[curr_reg]['min']
                                curr_max = region_stat[curr_reg]['max']
                                diff = min(abs(img[row, col] - curr_max), abs(img[row, col] - curr_min))
                                if diff < threshold:
                                    nb_list.append(curr_reg)
                                    nb_diff.append(diff)
                    if len(nb_list) == 0:
                        continue
                    else:
                        min_idx = np.argmin(nb_diff)
                        region[row, col] = nb_list[min_idx]
                        change += 1
        if change == 0:
            break
    return region

def region_split_merge(comp, log=False):
    reg_idx = np.max(comp) + 1
    height, width = comp.shape
    comp_tb = {}
    comp_h = {}
    h_list = []
    for row in range(height):
        reg_list = np.unique(comp[row]).tolist()
        reg_list.remove(0)
        for reg in reg_list:
            if reg not in comp_tb:
                comp_tb[reg] = [row, row]
            comp_tb[reg][1] = row
    for k, v in comp_tb.items():
        h = v[1] - v[0]
        comp_h[k] = h
        h_list.append(h)
    h_list.sort()
    # for k, v in comp_h.items():
    #     print(f"{k}: {v}")

    median = h_list[len(h_list)//2]
    mean = np.mean(h_list)

    # if log:
    #     print(h_list)
    #     print([h/median for h in h_list])
    #     print(h_list[(len(h_list)//2) - 1])
    #     print(h_list[len(h_list)//2])
    #     print(h_list[(len(h_list)//2) + 1])
    #     print(np.mean(h_list))

    # split tall region
    for k, v in comp_h.items():
        if v/mean > 1.8:
            # print(k)
            for row in range(comp_tb[k][0] + (v//2), comp_tb[k][1]+1):
                for col in range(width):
                    if comp[row, col] == k:
                        comp[row, col] = reg_idx
            comp_tb[reg_idx] = [comp_tb[k][0] + (v//2), comp_tb[k][1]]
            comp_tb[k][1] = comp_tb[k][0] + (v//2) - 1
            reg_idx += 1

    # merge region with similar row
    row = 0
    while row < height:
        # print(f"{row=}")
        if len(np.unique(comp[row])) > 2:
            reg_idxs = np.unique(comp[row]).tolist()
            reg_idxs.remove(0)
            a = reg_idxs[0]
            move_row_to = comp_tb[a][1]
            for j in range(1, len(reg_idxs)):
                b = reg_idxs[j]
                top = max(comp_tb[a][0], comp_tb[b][0])
                bottom = min(comp_tb[a][1], comp_tb[b][1])
                common_h = bottom - top
                a_ratio = common_h / (comp_tb[a][1] - comp_tb[a][0] + 1)
                b_ratio = common_h / (comp_tb[b][1] - comp_tb[b][0] + 1)
                merge_sub = a if a_ratio > b_ratio else b
                merge_main = b if a_ratio > b_ratio else a
                merge_ratio = a_ratio if a_ratio > b_ratio else b_ratio
                if merge_ratio > 0.7:
                    comp[comp == merge_sub] = merge_main
                    if comp_tb[merge_sub][1] < move_row_to:
                        move_row_to = comp_tb[merge_sub][1]
                    comp_tb.pop(merge_sub)
                a = merge_main
            if move_row_to > row:
                row = move_row_to
        row += 1

    # reindex region number from bottom to top
    reg_idx = [0]
    for row in range(height-1, -1, -1):
        for col in range(width):
            if comp[row, col] > 0:
                if comp[row, col] not in reg_idx:
                    reg_idx.append(comp[row, col])
    reg_map = {v:k for k, v in enumerate(reg_idx)}
    for row in range(height):
        for col in range(width):
            comp[row, col] = reg_map[comp[row, col]]
    return comp

def comp_pos(comp):
    height, width = comp.shape
    max_comp_value = np.max(comp)
    count = [0] * max_comp_value
    sum_h = [0] * max_comp_value
    sum_w = [0] * max_comp_value
    for row in range(height):
        for col in range(width):
            if comp[row, col] == 0:
                continue
            count[comp[row, col] - 1] += 1
            sum_h[comp[row, col] - 1] += row
            sum_w[comp[row, col] - 1] += col
    for i in range(max_comp_value):
        sum_h[i] /= count[i]
        sum_w[i] /= count[i]

    return sum_h, sum_w

def merge_small_region(region_img, threshold_scale=0.33):
    height, width = region_img.shape
    total_region = np.max(region_img)
    region_size = [0] * (total_region)
    for row in range(height):
        for col in range(width):
            if region_img[row, col] > 0:
                region_size[region_img[row, col] - 1] += 1
    # print(region_size)
    # print(np.mean(region_size))
    threshold = np.mean(region_size) * threshold_scale
    region_id_size = [[idx+1, v] for idx, v in enumerate(region_size)]
    region_id_size.sort(key=lambda x: x[1])
    to_map = {}
    curr_idx = len(region_id_size) - 1
    region_pos = comp_pos(region_img)
    for idx, size in region_id_size:
        if size > threshold:
            break
        curr_pos = region_pos[0][idx-1], region_pos[1][idx-1]
        # print(curr_pos)
        closest = [idx, 10000]
        for jdx, pos in enumerate(zip(*region_pos)):
            if jdx == idx-1:
                continue
            dist = np.sqrt((pos[0] - curr_pos[0])**2 + (pos[1] - curr_pos[1])**2)
            # print(jdx+1, dist, pos[0], pos[1])
            if dist < closest[1] and region_size[jdx] > threshold:
                closest = [jdx, dist]
        # print(f"{idx} -> {closest[0] + 1}")
        to_map[idx] = closest[0] + 1
    for a, b in to_map.items():
        region_img[region_img == a] = b
    region_img = region_reindex(region_img)
    return region_img


