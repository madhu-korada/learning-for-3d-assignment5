import numpy as np
import argparse

import torch
import pytorch3d
from models import cls_model
from utils import create_dir, viz_cls
from data_loader import get_data_loader

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    
    parser.add_argument('--task', type=str, default="cls", help='The task: cls or seg')
    
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model(num_classes=args.num_cls_class).to(args.device)
    
    # Load Model Checkpoint
    # model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    model_path = './checkpoints/cls/best_model.pt'
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy(np.load(args.test_label))

    # ------ TO DO: Make Prediction ------
    test_dataloader = get_data_loader(args=args, train=False)
    print ("successfully loaded data")
    
    # rotation
    rotation = False
    if rotation:
        rotation = torch.tensor([20, 0, 0])
        R = pytorch3d.transforms.euler_angles_to_matrix(rotation, 'XYZ')
        test_dataloader.dataset.data = (R @ test_dataloader.dataset.data.transpose(1, 2)).transpose(1, 2)
    
    correct_obj = 0
    num_obj = 0
    preds_labels_arr = []
    for batch in test_dataloader:
        point_clouds, labels = batch
        point_clouds = point_clouds[:, ind].to(args.device)
        labels = labels.to(args.device).to(torch.long)

        with torch.no_grad():
            pred_labels = model(point_clouds).argmax(dim=1, keepdim=False)
            # pred_labels = torch.argmax(model(point_clouds), dim=-1, keepdim=False)
        correct_obj += pred_labels.eq(labels.data).cpu().sum().item()
        num_obj += labels.size()[0]

        preds_labels_arr.append(pred_labels)

    accuracy = correct_obj / num_obj
    print(f"test accuracy: {accuracy}")
    preds_labels = torch.cat(preds_labels_arr).detach().cpu()
    # np.save(f"{args.output_dir}/preds_labels.npy", preds_labels)
    
    # find where all the predictions are wrong
    # wrong_preds, correct_preds = [], []
    # for i in range(len(test_label)):
    #     if test_label[i] != preds_labels[i]:
    #         wrong_preds.append(i)
    #     else:
    #         correct_preds.append(i)
    
    wrong_preds_idx = torch.argwhere(test_label != preds_labels)#.flatten()
    correct_preds_idx = torch.argwhere(test_label == preds_labels)
    
    print(f"Number of correct predictions: {wrong_preds_idx.shape}")
    print(f"Number of wrong predictions: {correct_preds_idx.shape}")
    
    
    num_gifs = 0
    for i in range(len(wrong_preds_idx)):
        # print(f"Wrong prediction at index {wrong_preds_idx[i]}")
        idx = wrong_preds_idx[i].item()
        point_cloud = test_dataloader.dataset.data[idx, ind].detach().cpu()
        gt_label = test_dataloader.dataset.label[idx].detach().cpu().item()
        pred_label = preds_labels[idx].detach().cpu().item()
        
        gif_path = f"{args.output_dir}/cls_wrong_pred_idx_{i}_gt_{gt_label}_pred_{pred_label}.gif"
        viz_cls(point_cloud, gif_path, args.device)
        
        num_gifs += 1
        if num_gifs >= 10:
            break
    
    num_gifs = 0
    for i in range(len(correct_preds_idx)):
        # print(f"Correct prediction at index {correct_preds_idx[i]}")
        idx = correct_preds_idx[i].item()
        point_cloud = test_dataloader.dataset.data[idx, ind].detach().cpu()
        gt_label = test_dataloader.dataset.label[idx].detach().cpu().item()
        pred_label = preds_labels[idx].detach().cpu().item()
        
        gif_path = f"{args.output_dir}/cls_correct_pred_idx_{i}_gt_{gt_label}_pred_{pred_label}.gif"
        viz_cls(point_cloud, gif_path, args.device)
    
        num_gifs += 1
        if num_gifs >= 10:
            break
    
        
    
    
