import json
import os

from IPython.display import display, HTML
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def traj2conn_json(graph_path, idx, trajectory_data):
    trajectory = trajectory_data[idx]
    instr_id = trajectory['instr_id']
    scan = instr_id2scan[instr_id]
    viewpointId2idx = {}
    with open(graph_path % scan) as f:
        conn_data = json.load(f)
    for i, item in enumerate(conn_data):
        viewpointId2idx[item['image_id']] = i
    return trajectory, viewpointId2idx, conn_data


def gen_conns(trajectory, viewpointId2idx, conn_data):
    node = conn_data[viewpointId2idx[trajectory['trajectory'][0][0]]]
    node = {k: v for k, v in node.items()}
    node['unobstructed'] = [False] * len(trajectory['trajectory'])
    conns = [node]
    prev_viewpoint = node['image_id']
    for n, (viewpoint, heading, elevation) in enumerate(trajectory['trajectory'][1:]):
        node = conn_data[viewpointId2idx[viewpoint]]
        node = {k: v for k, v in node.items()}
        prev_viewpoint = conns[-1]['image_id']
        if viewpoint != prev_viewpoint:
            assert node['unobstructed'][viewpointId2idx[prev_viewpoint]]
            node['unobstructed'] = [False] * len(trajectory['trajectory'])
            node['unobstructed'][len(conns) - 1] = True
            conns.append(node)
    return conns


def idx2scan_folder(idx, trajectory_data):
    trajectory = trajectory_data[idx]
    instr_id = trajectory['instr_id']
    print(instr_id, trajectory)
    scan = instr_id2scan[instr_id]
    txt = instr_id2txt[instr_id]
    return [scan, folders[scan]]


def build_dicts(trajectory_path, instruction_path):
    with open(trajectory_path) as f:
        trajectory_data = json.load(f)
    with open(instruction_path) as f:
        instruction_data = json.load(f)

    instr_id2txt = {f"{d['path_id']}_{n}": txt for d in instruction_data for n, txt in enumerate(d['instructions'])}
    instr_id2scan = {f"{d['path_id']}_{n}":
                         d['scan'] for d in instruction_data for n, txt in enumerate(d['instructions'])}
    scan2trajidx = {
        instr_id2scan[traj['instr_id']]: idx for idx, traj in enumerate(trajectory_data)}
    instr_id2trajidx = {traj['instr_id']: idx for idx, traj in enumerate(trajectory_data)}
    return trajectory_data, instruction_data, instr_id2txt, instr_id2scan, scan2trajidx, instr_id2trajidx


if __name__ == '__main__':
    # Get all folders in mesh
    instr_id = "224_1"
    folders = {}
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--remote-debugging-port=9222')
    executable_path = "/home/krishna/Projects/VLN-BEVBert/visualise_dir/chromedriver"
    # os.environ["PATH"] += os.pathsep + executable_path
    browser = webdriver.Chrome(options=chrome_options)
    browser.implicitly_wait(5)
    url = "http://10.18.65.188:8001/connectivity.html"
    with open("/home/krishna/Projects/VLN-BEVBert/datasets/R2R/connectivity/scans.txt") as txt:
        scans = txt.readlines()
    for name in os.listdir("/mnt/Storage1/Krishna/datasets/matterport/v1/scans/"):
        if name == 'data':
            continue
        subfolder = os.listdir(f"/mnt/Storage1/Krishna/datasets/matterport/v1/scans/{name}/matterport_mesh/")
        folders[name] = subfolder[0]

    trajectory_path = "/home/krishna/Projects/VLN-HAMT/test/baseline/preds/detail_test.json"

    instruction_path = "/home/krishna/Projects/VLN-BEVBert/datasets/R2R/annotations/R2R_test_enc.json"

    graph_path = "/home/krishna/Projects/VLN-BEVBert/datasets/R2R/connectivity/%s_connectivity.json"

    trajectory_data, instruction_data, instr_id2txt, instr_id2scan, scan2trajidx, instr_id2trajidx \
        = build_dicts(trajectory_path, instruction_path)

    idxs = [instr_id2trajidx[instr_id]]
    scan_folders = [idx2scan_folder(idx, trajectory_data) for idx in idxs]
    print(scan_folders)
    instr_id = trajectory_data[idxs[0]]['instr_id']
    print(instr_id)

    # show instructions

    instruction = instr_id2txt[instr_id]
    print(instruction)
    print('')
    for i in ['0','1','2']:
        print(instr_id2txt[instr_id[:-1]+i])

    scan_folders = [idx2scan_folder(idx, trajectory_data) for idx in idxs]

    with open('./jolin_mesh_names.json', 'w') as fp:
        json.dump(scan_folders, fp)

    for idx, (scan, folder) in zip(idxs, scan_folders):
        with open('./%s.json'% scan, 'w') as fp:
            trajectory, viewpointId2idx, conn_data = traj2conn_json(graph_path, idx, trajectory_data)
            trajectory_ = trajectory_data[idx]
            json.dump(gen_conns(trajectory_, viewpointId2idx, conn_data), fp)
    browser.get(url) #navigate to the page
    imgData = browser.execute_script('return renderer.domElement.toDataURL().replace("image/png", '
                                     '"image/octet-stream")')
    # display(HTML(f'''<img src="{imgData}">'''))
