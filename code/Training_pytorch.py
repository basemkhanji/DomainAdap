#!/usr/bin/env python
# coding: utf-8
# Many thanks for christoph.hasse@cern.ch & niklas.nolte@cern.ch for their orginal work and allowing me to share!

import argparse
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from model import BaselineModel


def write_epm_file(preds, truth, epm_fname):
    import uproot3

    pred_tags = preds.squeeze()

    epm_tags = np.where(pred_tags > 0.5, 1, -1).astype(np.int32)
    true_id = np.where(truth.squeeze() == 0, -511, 511).astype(np.int32)
    eta = np.where(pred_tags > 0.5, 1 - pred_tags, pred_tags)

    with uproot3.recreate(f"{epm_fname}.root", compression=None) as file:
        file["DecayTree"] = uproot3.newtree({"B_TRUEID": np.int32, "tag": np.int32, "eta": np.float64})
        t = file["DecayTree"]
        t["B_TRUEID"].newbasket(true_id)
        t["tag"].newbasket(epm_tags)
        t["eta"].newbasket(eta)


def train_model(files, model_out_name, scaler_out_name, n_epochs, train_frac, batch_size, make_epm_output):

    print("Starting Training")
    # some torch setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(25031992)
    torch.cuda.manual_seed(25031992)

    features = np.concatenate([f["features"] for f in files])
    tags = np.concatenate([f["B_TRUEID"] for f in files]).reshape((-1, 1))
    tags = np.where(tags == 521, 1, 0).astype(np.int32)

    evt_borders = files[0]["evt_borders"]
    for f in files[1:]:
        evt_borders = np.concatenate((evt_borders, f["evt_borders"][1:] + evt_borders[-1]))

    assert evt_borders[-1] == len(features)

    # probnnmu has a bin at -1, for particles that don't have muon info
    # map that to 0
    features[features[:, 3] == -1, 3] = 0

    # scale data, and safe scaler for later use
    scaler = RobustScaler()
    features = scaler.fit_transform(features)
    joblib.dump(scaler, f"{scaler_out_name}.bin")

    borders = np.array(list(zip(evt_borders[:-1], evt_borders[1:])))
    idx_vec = np.zeros(len(features), dtype=np.int64)
    for i, (b, e) in enumerate(borders):
        idx_vec[b:e] = i

    evt_split = int(len(borders) * train_frac)

    track_split = evt_borders[evt_split]

    train_tags = torch.tensor(tags[:evt_split], dtype=torch.float32).to(device)
    train_feat = torch.tensor(features[:track_split]).to(device)
    train_idx = torch.tensor(idx_vec[:track_split]).to(device)

    test_tags_np = tags[evt_split:]
    test_tags = torch.tensor(test_tags_np, dtype=torch.float32).to(device)
    test_feat = torch.tensor(features[track_split:]).to(device)
    test_idx = torch.tensor(idx_vec[track_split:]).to(device)

    train_borders = [(x[0, 0], x[-1, 1]) for x in np.array_split(borders[:evt_split], len(borders[:evt_split]) // batch_size)]
    test_borders = [
        (x[0, 0], x[-1, 1])
        for x in np.array_split(borders[evt_split:] - borders[evt_split][0], len(borders[evt_split:]) // batch_size)
    ]

    model = BaselineModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-5, patience=5)

    all_tl = []
    all_vl = []
    all_ac = []

    mypreds = np.zeros((len(test_tags), 1))

    for epoch in range(n_epochs):
        model.train()
        trainloss = 0
        for batch_idx, (beg, end) in enumerate(train_borders):
            optimizer.zero_grad()

            data = train_feat[beg:end]
            idx = train_idx[beg:end] - train_idx[beg]
            e_beg, e_end = train_idx[[beg, end - 1]]
            # one past the last event is the boundary
            e_end += 1
            target = train_tags[e_beg:e_end]

            output = model(data, idx)
            loss = nn.functional.binary_cross_entropy_with_logits(output, target)

            loss.backward()

            optimizer.step()

            trainloss += loss.detach().cpu().numpy()

        # averaged trainloss of epoch
        all_tl.append(trainloss / (batch_idx + 1))
        trainloss = 0

        model.eval()
        valloss = 0
        for batch_idx, (beg, end) in enumerate(test_borders):

            data = test_feat[beg:end]
            # indices for the index_add inside the forward()
            idx = test_idx[beg:end] - test_idx[beg]

            # minus to make the test_idx start at 0 since we are indexing into
            # the split off test_tags array
            e_beg, e_end = test_idx[[beg, end - 1]] - test_idx[0]
            # one past the last event is the boundary
            e_end += 1
            target = test_tags[e_beg:e_end]

            with torch.no_grad():
                output = model(data, idx)

            mypreds[e_beg:e_end] = torch.sigmoid(output.detach()).cpu().numpy()
            valloss += nn.functional.binary_cross_entropy_with_logits(output, target).detach().cpu().numpy()

        acc = np.mean((mypreds > 0.5) == test_tags_np)
        all_vl.append(valloss / (batch_idx + 1))
        all_ac.append(acc)

        scheduler.step(valloss / (batch_idx + 1))

        print(
            f"Epoch: {epoch}/{n_epochs} | Val loss {valloss/(batch_idx+1):.5f} | AUC: {roc_auc_score(test_tags_np, mypreds):.5f} | ACC: {acc:.5f}",
            end="\r",
        )

    print("Training complete")
    print(f"Minimum training loss: {min(all_vl):.5f} in epoch: {np.argmin(all_vl)}")
    print(f"Maximum training ACC:  {max(all_ac):.5f} in epoch: {np.argmax(all_ac)}")

    # done training so let's set it to eval
    model.eval()

    torch.save(model.state_dict(), f"{model_out_name}.pt")

    if make_epm_output:
        print("Writing output for EPM")
        try:
            write_epm_file(mypreds, test_tags_np, f"{model_out_name}_epm")
        except ImportError:
            print("Option make-epm-output requires uproot3 package to be available.\n Writing of EPM output skipped!")

    print("Making plots.")
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"font.size": 22})

    plt.figure(figsize=(16, 9))
    plt.plot(all_tl, label="Train Loss")
    plt.plot(all_vl, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylim(0.6, 0.8)
    plt.grid()
    plt.savefig("Loss_vs_Epoch.png")


def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} not a floating-point literal")

    if x <= 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x} not in range (0.0, 1.0]")

    return x


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Model for Flavour Tagging.")
    parser.add_argument("filenames", nargs="+", help="Files that contain training data. *.npz files expected)")
    parser.add_argument(
        "-model-out-name",
        default="model",
        help="File name to save weights into. Default is model.pt",
    )
    parser.add_argument(
        "-scaler-out-name",
        default=None,
        help="File name to save scaler into. Default is MODELNAME_scaler.bin",
    )
    parser.add_argument("-epochs", dest="n_epochs", default=1000, type=int, help="Batch size")
    parser.add_argument(
        "-train-frac",
        default=0.75,
        type=restricted_float,
        help="Fraction of data to use for training",
    )
    parser.add_argument("-batch-size", default=1000, type=int, help="Batch size")
    parser.add_argument("--make-epm-output", action="store_false", help="Write tagged validataion data into root file for EPM")

    args = parser.parse_args()

    files = [np.load(f) for f in args.filenames]

    if args.scaler_out_name == None:
        args.scaler_out_name = args.model_out_name + "_scaler"

    train_model(
        files, args.model_out_name, args.scaler_out_name, args.n_epochs, args.train_frac, args.batch_size, args.make_epm_output
    )
