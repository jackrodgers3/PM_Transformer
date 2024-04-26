import os
from timeit import default_timer as timer
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import math
from math import pi
import torch
from tqdm import tqdm
import random
import pickle
from scipy.spatial import distance
import pandas as pd
temp_data_file = r"D:\Data\Research\PUPPI\ZJetsToQQ_HT-800toInf/"
np.random.seed(23)


def gen_dataframe(rfilename, num_event, num_start=0):
    """
        select pfcands from original root and convert to a pandas dataframe.
        Returned is a list of dataframes, with one dataframe for one event.
        """
    print(f"reading events from {num_start} to {num_start + num_event}")
    tree = uproot.open(rfilename)["Events"]
    pfcands = tree.arrays(filter_name="PF_*", entry_start = num_start,
                          entry_stop = num_event + num_start)
    genparts = tree.arrays(filter_name="packedGenPart_*",
                           entry_start = num_start, entry_stop = num_event + num_start)
    print(f"# of entries in file: {tree.num_entries}")
    num_entries = tree.num_entries
    df_pf_list = []
    df_gen_list = []

    for i in range(num_event):
        event = pfcands[i]
        selected_features = ['PF_eta', 'PF_phi', 'PF_pt',
                             'PF_pdgId', 'PF_charge', 'PF_puppiWeight', 'PF_puppiWeightNoLep', 'PF_dz',
                             'PF_fromPV'
                             ]
        pf_chosen = event[selected_features]
        df_pfcands = ak.to_dataframe(pf_chosen)
        df_pfcands = df_pfcands[abs(df_pfcands['PF_eta']) < 2.5]
        df_pf_list.append(df_pfcands)

    for i in range(num_event):
        event = genparts[i]
        selected_features = ['packedGenPart_eta', 'packedGenPart_phi',
                             'packedGenPart_pt', 'packedGenPart_pdgId', 'packedGenPart_charge']
        gen_chosen = event[selected_features]
        df_genparts = ak.to_dataframe(gen_chosen)
        selection = (df_genparts['packedGenPart_eta'] < 2.5) & (abs(df_genparts['packedGenPart_pdgId']) != 12) & \
                    (abs(df_genparts['packedGenPart_pdgId']) != 14) & (abs(df_genparts['packedGenPart_pdgId']) != 16)
        df_genparts = df_genparts[selection]
        df_gen_list.append(df_genparts)

    return df_pf_list, df_gen_list


def prepare_dataset(rfilename, num_event, dR, num_start = 0):
    data_list = []
    df_pf_list, df_gen_list = gen_dataframe(rfilename, num_event, num_start)

    PTCUT = 0

    for num in tqdm(range(len(df_pf_list))):

        df_pfcands = df_pf_list[num]
        LV_index = np.where((df_pfcands['PF_puppiWeight'] > 0.99) & (df_pfcands['PF_charge'] != 0) &
                            (df_pfcands['PF_pt'] > PTCUT) & (df_pfcands['PF_fromPV'] > 2))[0]
        PU_index = np.where((df_pfcands['PF_puppiWeight'] < 0.01) & (df_pfcands['PF_charge'] != 0) &
                            (df_pfcands['PF_pt'] > PTCUT) & (df_pfcands['PF_fromPV'] < 1))[0]
        if LV_index.shape[0] < 5 or PU_index.shape[0] < 50:
            continue
        Neutral_index = np.where(df_pfcands['PF_charge'] == 0)[0]
        Charge_index = np.where(df_pfcands['PF_charge'] != 0)[0]

        #label samples
        label = df_pfcands.loc[:, ['PF_puppiWeight']].to_numpy()
        label = torch.from_numpy(label).view(-1)
        label = label.type(torch.long)

        node_features = df_pfcands.drop(df_pfcands.loc[:, ['PF_charge']], axis=1).drop(
            df_pfcands.loc[:, ['PF_fromPV']], axis=1).to_numpy()

        node_features = torch.from_numpy(node_features)
        node_features = node_features.type(torch.float32)

        # set the charge pdgId for one hot encoding later
        # ToDO: fix for muons and electrons
        index_pdgId = 3
        node_features[[Charge_index.tolist()], index_pdgId] = 0
        # get index for mask training and testing samples for pdgId(3) and puppiWeight(4)
        # one hot encoding for pdgId and puppiWeight
        pdgId = node_features[:, index_pdgId]
        photon_indices = (pdgId == 22)
        pdgId[photon_indices] = 1
        hadron_indices = (pdgId == 130)
        pdgId[hadron_indices] = 2
        pdgId = pdgId.type(torch.long)
        # print(pdgId)
        pdgId_one_hot = torch.nn.functional.one_hot(pdgId)
        pdgId_one_hot = pdgId_one_hot.type(torch.float32)
        assert pdgId_one_hot.shape[1] == 3, "pdgId_one_hot.shape[1] != 3"
        # print ("pdgID_one_hot", pdgId_one_hot)
        # set the neutral puppiWeight to default
        index_puppi = 4
        pWeight = node_features[:, index_puppi].clone()
        node_features[[Neutral_index.tolist()], index_puppi] = 2
        puppiWeight = node_features[:, index_puppi]
        puppiWeight = puppiWeight.type(torch.long)
        puppiWeight_one_hot = torch.nn.functional.one_hot(puppiWeight)
        puppiWeight_one_hot = puppiWeight_one_hot.type(torch.float32)
        index_puppi_chg = 5
        pWeightChg = node_features[:, index_puppi_chg].clone()
        # columnsNamesArr = df_pfcands.columns.values
        node_features = torch.cat(
            (node_features[:, 0:3], pdgId_one_hot, puppiWeight_one_hot), 1)
        #    (node_features[:, 0:3], pdgId_one_hot, node_features[:, -1:], puppiWeight_one_hot), 1)
        # i(node_features[:, 0:3], pdgId_one_hot,node_features[:,5:6], puppiWeight_one_hot), 1)
        # (node_features[:, 0:4], pdgId_one_hot, puppiWeight_one_hot), 1)

        if num == -1:
            print("pdgId dimensions: ", pdgId_one_hot.shape)
            print("puppi weights dimensions: ", puppiWeight_one_hot.shape)
            print("last dimention: ", node_features[:, -1:].shape)
            print("node_features dimension: ", node_features.shape)
            print("node_features[:, 0:3] dimention: ",
                  node_features[:, 0:3].shape)
            print("node_features dimension: ", node_features.shape)
            print("node_features[:, 6:7]",
                  node_features[:, 6:7].shape)  # dz values
            # print("columnsNamesArr", columnsNamesArr)
            # print ("pdgId_one_hot " , pdgId_one_hot)
            # print("node_features[:,-1:]",node_features[:,-1:])
            # print("puppi weights", puppiWeight_one_hot)
            # print("node features: ", node_features)

        # node_features = node_features.type(torch.float32)
        # construct edge index for graph

        phi = df_pfcands['PF_phi'].to_numpy().reshape((-1, 1))
        eta = df_pfcands['PF_eta'].to_numpy().reshape((-1, 1))

        df_gencands = df_gen_list[num]
        gen_features = df_gencands.to_numpy()
        gen_features = torch.from_numpy(gen_features)
        gen_features = gen_features.type(torch.float32)

        dist_phi = distance.cdist(phi, phi, 'cityblock')
        # deal with periodic feature of phi
        indices = np.where(dist_phi > pi)
        temp = np.ceil((dist_phi[indices] - pi) / (2 * pi)) * (2 * pi)
        dist_phi[indices] = dist_phi[indices] - temp
        dist_eta = distance.cdist(eta, eta, 'cityblock')

        dist = np.sqrt(dist_phi ** 2 + dist_eta ** 2)
        edge_source = np.where((dist < dR) & (dist != 0))[0]
        edge_target = np.where((dist < dR) & (dist != 0))[1]

        edge_index = np.array([edge_source, edge_target])
        edge_index = torch.from_numpy(edge_index)
        edge_index = edge_index.type(torch.long)
        data_list.append([node_features, label])
    return data_list


def main():
    folder = temp_data_file
    rfiles = os.listdir(folder)
    # tvt = (0.7, 0.15, 0.15)
    dR = 0.4

    for i in range(len(rfiles)):
        start = timer()

        iname = folder + '/output_' + str(i+1) + '.root'
        num_events_train = 14000
        oname = folder + '/train_dat'+str(i+1)+'.pkl'
        dataset_train = prepare_dataset(iname, num_events_train, dR)
        with open(oname, 'wb') as fp:
            pickle.dump(dataset_train, fp)
        fp.close()

        num_events_test = 3000
        oname = folder + '/test_dat'+str(i+1)+'.pkl'
        dataset_test = prepare_dataset(iname, num_events_test, dR, num_events_train)
        with open(oname, 'wb') as fp:
            pickle.dump(dataset_test, fp)
        fp.close()

        num_events_valid = 3000
        oname = folder + '/valid_dat'+str(i+1)+'.pkl'
        dataset_valid = prepare_dataset(iname, num_events_valid, dR, num_events_train+num_events_test)
        with open(oname, 'wb') as fp:
            pickle.dump(dataset_valid, fp)
        fp.close()

        end = timer()
        program_time = end - start
        print(f"Build time: {str(program_time)}")


if __name__ == '__main__':
    main()
